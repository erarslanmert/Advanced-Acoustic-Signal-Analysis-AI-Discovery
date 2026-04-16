import re
import os
import numpy as np
from datetime import datetime, timedelta

def hex_to_int(hex_str):
    if not hex_str:
        return 0
    try:
        return int(hex_str, 16)
    except:
        return 0

class AMMSTiming:
    def __init__(self, samples, times, audio_start_sample, log_fs, wav_fs=None, log_metadata_map=None, anchor_source=None, signature_rate=None):
        """
        samples: list/array of absolute sample numbers from sync points
        times: list/array of absolute microseconds from sync points
        audio_start_sample: the sample index where audio file begins (0-indexed in log)
        log_fs: sampling rate stated in the log (usually 16000)
        wav_fs: actual sampling rate of the WAV file (e.g. 48000)
        log_metadata_map: dict mapping sample numbers to the full line or dict of line fields
        anchor_source: The label (e.g. 'ptime') used as the absolute UTC anchor
        signature_rate: The measured hardware frequency (Approach 1)
        """
        self.anchor_source = anchor_source
        self.log_fs = float(log_fs)
        self.wav_fs = float(wav_fs) if wav_fs else self.log_fs
        
        # v32.61: If signature_rate is provided, we use it as our absolute truth for the hardware clock
        if signature_rate:
            self.true_log_fs = float(signature_rate)
        else:
            self.true_log_fs = self.log_fs
            
        self.samples = np.array(samples, dtype=np.float64)
        self.times = np.array(times, dtype=np.float64)
        self.audio_start_sample = float(audio_start_sample)
        
        # Resampling Scale Factor.
        self.rate_scale = self.wav_fs / self.log_fs
        
        # The 'True' effective frequency in the WAV file domain
        self.true_wav_fs = self.true_log_fs * self.rate_scale
        
        # Align samples to the start of the file (sample 0) and SCALE to file rate
        self.samples_corrected = (self.samples - self.audio_start_sample) * self.rate_scale
        
        # We continue to use the header-fs for nominal calculations, but the true_fs for corrections
        self.fs = self.wav_fs
        
        self.log_metadata = log_metadata_map if log_metadata_map else {}

        # Filter out invalid (decreasing) time syncs if many points exist
        if len(self.samples_corrected) > 1:
            diffs = np.diff(self.times)
            if np.any(diffs <= 0):
                valid_mask = np.ones(len(self.times), dtype=bool)
                for i in range(1, len(self.times)):
                    if self.times[i] <= self.times[i-1]:
                        valid_mask[i] = False
                self.samples_corrected = self.samples_corrected[valid_mask]
                self.times = self.times[valid_mask]

    def sample2time(self, sample_nr_rel):
        """
        Converts a relative sample number to a datetime.
        Uses signature-aware extrapolation for Sample 0 detection.
        """
        if len(self.times) == 0:
            return None

        if len(self.times) == 1:
            # Single point fallback: uses the TRUE measured frequency for extrapolation
            offset_s = (sample_nr_rel - self.samples_corrected[0]) / self.true_wav_fs
            us = self.times[0] + (offset_s * 1e6)
        else:
            # Linear interpolation (for points between locks)
            # np.interp also extrapolates outside the range using the first/last slope.
            # However, for highest precision, we calculate the projection manually if outside range.
            if sample_nr_rel < self.samples_corrected[0]:
                offset_s = (sample_nr_rel - self.samples_corrected[0]) / self.true_wav_fs
                us = self.times[0] + (offset_s * 1e6)
            else:
                us = np.interp(sample_nr_rel, self.samples_corrected, self.times)
        
        dt = datetime(1970, 1, 1) + timedelta(microseconds=us)
        return dt
    def time2sample(self, dt):
        """
        Converts a datetime back to a relative sample number.
        """
        if len(self.times) == 0:
            return 0.0
            
        epoch = datetime(1970, 1, 1)
        target_us = (dt - epoch).total_seconds() * 1e6
        
        if len(self.times) == 1:
            offset_us = target_us - self.times[0]
            sample_rel = self.samples_corrected[0] + (offset_us / 1e6 * self.true_wav_fs)
        else:
            if target_us < self.times[0]:
                offset_us = target_us - self.times[0]
                sample_rel = self.samples_corrected[0] + (offset_us / 1e6 * self.true_wav_fs)
            else:
                sample_rel = np.interp(target_us, self.times, self.samples_corrected)
            
        return float(sample_rel)

    def get_metadata_for_sample(self, sample_nr_rel):
        """Returns the log metadata for the sync point closest to the given sample."""
        if not self.log_metadata:
            return None
        
        # Find absolute sample in log
        abs_sample = sample_nr_rel + self.audio_start_sample
        
        # Find closest sync sample
        known_samples = list(self.log_metadata.keys())
        closest = min(known_samples, key=lambda x: abs(x - abs_sample))
        return self.log_metadata[closest]

def parse_amms_log(log_path, wav_sr=None):
    """
    Parses an AMMS .LOG file to extract timing vectors.
    wav_sr: If provided, scales the hardware sample indices to match this rate.
    Returns AMMSTiming object or None if no audio start/sync found.
    """
    if not os.path.exists(log_path):
        return None

    audio_start_sample = None
    try:
        raw_sync_samples = []
        raw_sync_times = []
        raw_sync_labels = []
        sync_metadata = {}
        
        # v32.95: Hard-Default to 16kHz (Standard AMMS Sensor Rate)
        # We ignore 'rate:' tags that match the WAV rate (e.g. 48k) to keep pipelines discrete.
        log_rate = 16000
        max_seen_rate = 16000

        # v32.22: Relaxed regex for better discovery (4+ digits)
        TS_REGEX = re.compile(r"(\w+):([0-9A-Fa-f]{4,});")

        with open(log_path, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # 1. Look for Audio Start (Canonical beginning)
                if "[AUDIO START]" in line:
                    m = re.search(r"sample:([0-9A-Fa-f]+);", line)
                    if m: audio_start_sample = hex_to_int(m.group(1))
                    m = re.search(r"rate:([0-9A-Fa-f]+);", line)
                    if m: 
                        r_val = hex_to_int(m.group(1))
                        # Only accept if it looks like a hardware sensor rate (not 48k audio)
                        if r_val != 48000: max_seen_rate = r_val
                
                # 2. Look for Sync points
                m_s = re.search(r"sample:([0-9A-Fa-f]+);", line)
                if not m_s: continue
                
                s_val = hex_to_int(m_s.group(1))
                t_val = None
                t_label = None
                
                # Find all time-like fields on this line
                found_fields = {}
                for f_match in TS_REGEX.finditer(line):
                    found_fields[f_match.group(1)] = f_match.group(2)
                
                # Priority mapping for time labels
                for label in ["ptime", "tsyncTime", "time", "tsync"]:
                    if label in found_fields:
                        t_val = hex_to_int(found_fields[label])
                        t_label = label
                        break
                
                # v32.70: Continuous Rate Discovery
                if "rate" in found_fields:
                    max_seen_rate = hex_to_int(found_fields["rate"])
                
                if t_val is not None:
                    # v32.42: Automatic Human-Era Scaler 
                    # Detects unit mismatch (e.g. 10ns ticks vs microseconds)
                    # 2.0e15 is the year 2033 in microseconds. 
                    # If value is significantly larger, it's likely higher resolution.
                    while t_val > 2.0e15:
                        t_val //= 10
                    
                    # Epoch adjustment for tsyncTime (year 0 vs 1970)
                    if t_label == "tsyncTime":
                        # tsyncTime is seconds since 0001-01-01. Offset to 1970-01-01.
                        # Using 62167219200000000 us (Standard Unix Epoch Delta)
                        offset_us = 719528 * 24 * 3600 * 1000000
                        t_val -= offset_us
                    
                    raw_sync_samples.append(s_val)
                    raw_sync_times.append(t_val)
                    raw_sync_labels.append(t_label)
                    
                    # Store metadata
                    fields = {}
                    for field_match in re.finditer(r"(\w+):([0-9A-Fa-f]+);", line):
                        fields[field_match.group(1)] = field_match.group(2)
                    fields['_raw'] = line
                    fields['_source'] = t_label
                    sync_metadata[s_val] = fields
        unique_samples = []
        unique_times = []
        unique_labels = []
        seen_samples = set()
        
        # We also need to keep metadata in sync
        dedup_metadata = {}
        
        raw_rows = zip(raw_sync_samples, raw_sync_times, raw_sync_labels)
        for s, t, l in raw_rows:
            if s not in seen_samples:
                unique_samples.append(s)
                unique_times.append(t)
                unique_labels.append(l)
                seen_samples.add(s)
                if s in sync_metadata:
                    dedup_metadata[s] = sync_metadata[s]
        
        raw_sync_samples = unique_samples
        raw_sync_times = unique_times
        raw_sync_labels = unique_labels
        sync_metadata = dedup_metadata

        if audio_start_sample is None:
            if raw_sync_samples:
                audio_start_sample = raw_sync_samples[0]
            else:
                return None

        # v32.75: Strict De-Duplication of Log points
        unique_samples = []
        unique_times = []
        unique_labels = []
        seen = set()
        for i in range(len(raw_sync_samples)):
            s = raw_sync_samples[i]
            if s not in seen:
                unique_samples.append(s)
                unique_times.append(raw_sync_times[i])
                unique_labels.append(raw_sync_labels[i])
                seen.add(s)
        
        raw_sync_samples = unique_samples
        raw_sync_times = unique_times
        raw_sync_labels = unique_labels

        # ─────────────────────────────────────────────────────────────
        # v32.40: Exhaustive Master Anchor Discovery
        # ─────────────────────────────────────────────────────────────
        # 1. First, find all absolute candidates
        abs_candidates = []
        for i in range(len(raw_sync_times)):
            t = raw_sync_times[i]
            l = raw_sync_labels[i]
            s = raw_sync_samples[i]
            
            # Print trace for the first few points to help diagnose
            if i < 5 or i == len(raw_sync_times) - 1:
                print(f"Auto-Sync: TRACE -> Point[{i}] Sample:{s}, Time:{t}, Label:{l}")
                
            if l in ['ptime', 'tsyncTime'] or t > 1.0e13:
                abs_candidates.append(i)

        # 1. Selection and Signature Calculation (Approach 1)
        master_anchor = None # (sample, absolute_us, label)
        signature_rate = log_rate # Default fallback to nominal
        
        if abs_candidates:
            # v32.41: Magnitude-Validated Anchor Selection
            actual_abs_idx = None
            for idx in abs_candidates:
                if raw_sync_times[idx] > 1.0e13:
                    actual_abs_idx = idx
                    break
            
            if actual_abs_idx is not None:
                master_anchor = (raw_sync_samples[actual_abs_idx], raw_sync_times[actual_abs_idx], raw_sync_labels[actual_abs_idx])
                
                # v32.95: Accurate Signature Extraction
                # Only calculate rate between two valid ABSOLUTE locks (> 1e13).
                # This prevents comparing Uptime (1970) to GPS (2025) which creates a 0.2Hz rate.
                valid_abs = [cid for cid in abs_candidates if raw_sync_times[cid] > 1.0e13]
                if len(valid_abs) >= 2:
                    idx1 = valid_abs[0]
                    idx2 = valid_abs[-1]
                    ds = raw_sync_samples[idx2] - raw_sync_samples[idx1]
                    dt = (raw_sync_times[idx2] - raw_sync_times[idx1]) / 1.0e6
                    if dt > 0 and ds > 0:
                        signature_rate = ds / dt
                        print(f"Auto-Sync: Physical Hardware Rate -> {signature_rate:.4f} Hz")
                    else:
                        signature_rate = float(max_seen_rate)
                else:
                    signature_rate = float(max_seen_rate)
                    print(f"Auto-Sync: Nominal Rate -> {signature_rate:.4f} Hz")
        
        final_sync_times = np.array(raw_sync_times, dtype=np.float64)
        if master_anchor is not None:
            # v32.65: Two-Pass History Repair (Eliminates 1970 Epoch Teleportation)
            # We initialize the 'rolling anchor' with our best absolute ground truth.
            # This ensures that early points (before the lock) are back-projected 
            # into the human era immediately.
            m_s, m_t, _ = master_anchor
            curr_anchor_s = m_s
            curr_anchor_t = m_t
            
            normalized_times = []
            for i in range(len(raw_sync_times)):
                s_curr = raw_sync_samples[i]
                t_curr = raw_sync_times[i]
                l_curr = raw_sync_labels[i]
                
                # v32.97: Strict Magnitude Filter
                # Ignore 'ptime' labels if the value is just uptime (< 2010 AD).
                is_curr_abs = (t_curr > 1.0e13)
                
                if is_curr_abs:
                    curr_anchor_s = s_curr
                    curr_anchor_t = t_curr
                    normalized_times.append(t_curr)
                else:
                    # Anchor to either the previous lock OR the first future lock (master_anchor)
                    dt_samples = s_curr - curr_anchor_s
                    dt_us = (dt_samples / signature_rate) * 1e6
                    normalized_times.append(curr_anchor_t + dt_us)
            
            final_sync_times = np.array(normalized_times)
            if len(normalized_times) > 0:
                print(f"Auto-Sync: First sync point back-projected to -> {datetime(1970, 1, 1) + timedelta(microseconds=normalized_times[0])}")
            
            # Print Final Summary using Signature Rate from Master Anchor for Sample 0
            m_anchor_s, m_anchor_t, m_label = master_anchor
            start_absolute = m_anchor_t - (m_anchor_s - audio_start_sample) / signature_rate * 1e6
            start_dt = datetime(1970, 1, 1) + timedelta(microseconds=start_absolute)
            print(f"Auto-Sync: File Start Reconciled (Signature Mode) -> {start_dt}")
            anchor_lbl = m_label
        else:
            print("Auto-Sync: WARNING - No Absolute Anchor found in log. Timeline will remain Relative.")
            anchor_lbl = None
        
        # Anchor reporting for UI already handled in the sweep block above.
        
        # Add signature info to metadata for first point to help trace
        if raw_sync_samples:
            sync_metadata[raw_sync_samples[0]]['_sig_rate'] = signature_rate
            sync_metadata[raw_sync_samples[0]]['_log_rate'] = max_seen_rate

        return AMMSTiming(raw_sync_samples, final_sync_times, audio_start_sample, max_seen_rate, wav_sr, sync_metadata, anchor_source=anchor_lbl, signature_rate=signature_rate)

    except Exception as e:
        print(f"Error parsing AMMS log {log_path}: {e}")
        return None
