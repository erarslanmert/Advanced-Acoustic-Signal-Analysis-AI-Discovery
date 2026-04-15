import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# We need QApplication to create Qt widgets natively for the GUI test
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt

# Import the core Annotator app
from AnnotatorApp import AudioAnnotatorApp

app_instance = None

def get_app():
    global app_instance
    if app_instance is None:
        app_instance = QApplication(sys.argv)
    return app_instance

class TestAudioAnnotatorApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qapp = get_app()
        
    def setUp(self):
        # Mock sounddevice completely during setup to avoid audio hardware conflicts
        self.patcher_sd = patch("AnnotatorApp.sd")
        self.mock_sd = self.patcher_sd.start()
        
        # Mock QFileDialog to avoid popup block traces
        self.patcher_qmsg = patch("AnnotatorApp.QMessageBox.critical")
        self.mock_qmsg = self.patcher_qmsg.start()
        
        # Instantiate the main window application natively
        self.app = AudioAnnotatorApp()
        
        # Load dummy audio dataset mimicking the ESG Bavaria 2015 layout
        self.app.sample_rate = 16000
        self.app.audio_data = np.zeros((16000, 1)) # 1 sec natively spanning 2D layout matrix
        self.app.filename = "dummy_audio.wav"
        
        # Manually initialize the mathematical Matplotlib bounds
        self.app.ax.set_xlim(0, 1.0)
        self.app.ax.set_ylim(-1, 1)

    def tearDown(self):
        self.patcher_sd.stop()
        self.patcher_qmsg.stop()
        self.app.close()
        
    @patch("AnnotatorApp.AnnotationEditDialog")
    def test_add_annotation(self, mock_dlg_class):
        # Setup mock behavior for the custom dialog
        mock_dlg = MagicMock()
        mock_dlg.exec_.return_value = 1 # QDialog.Accepted
        mock_dlg.class_label = "drone"
        mock_dlg.weight = 5.0
        mock_dlg_class.return_value = mock_dlg
        
        # Simulate user physically dragging across time [0.2s - 0.5s]
        self.app.on_select_interval(0.2, 0.5)
        
        # Verify the underlying annotation array appended securely
        self.assertEqual(len(self.app.annotations), 1)
        self.assertEqual(self.app.tree.topLevelItemCount(), 1)
        
        ann = self.app.annotations[0]
        self.assertEqual(ann["label"], "drone")
        self.assertEqual(ann["weight"], 5.0)
        
        # Verify the dynamically generated row text matching the UI output
        item = self.app.tree.topLevelItem(0)
        self.assertEqual(item.text(2), "drone")
        self.assertEqual(item.text(3), "1") # Row Number
        self.assertEqual(item.text(5), "5.00") # Weight Column (column 5)
        
    @patch("AnnotatorApp.AnnotationEditDialog")
    @patch("AnnotatorApp.QMessageBox.question")
    def test_delete_annotation_and_ripple_row(self, mock_question, mock_dlg_class):
        # Setup mock behavior for the custom dialog
        mock_dlg = MagicMock()
        mock_dlg.exec_.return_value = 1 
        mock_dlg_class.return_value = mock_dlg

        # Add two sequential annotations
        mock_dlg.class_label = "dog"
        self.app.on_select_interval(0.1, 0.2)
        
        mock_dlg.class_label = "cat"
        self.app.on_select_interval(0.3, 0.4)
        
        self.assertEqual(len(self.app.annotations), 2)
        
        # Mock user clicking "Yes" onto the native OS deletion dialog block!
        mock_question.return_value = QMessageBox.Yes
        
        # Select the first physical row ("dog") and trigger the deletion
        item = self.app.tree.topLevelItem(0)
        self.app.tree.setCurrentItem(item)
        self.app.delete_annotation(specific_item=item)
        
        # Verify "dog" is mathematically sliced out, 1 remaining annotation securely
        self.assertEqual(len(self.app.annotations), 1)
        self.assertEqual(self.app.annotations[0]["label"], "cat")
        
        # Verify UI ripple row effect (the former row 2 "cat" is now safely mapped to row 1!)
        remaining_item = self.app.tree.topLevelItem(0)
        self.assertEqual(remaining_item.text(3), "1")
        
    def test_pause_resume_toggle(self):
        # Fire up the audio explicitly simulating the playback event 
        self.app.playback_timer.start(50)
        self.assertTrue(self.app.playback_timer.isActive())
        self.assertFalse(self.app.paused)
        
        # Trigger Ctrl+Right Click (Pause Mode)
        self.app.toggle_pause()
        self.assertTrue(self.app.paused)                               # Flag locks properly
        self.assertFalse(self.app.playback_timer.isActive())           # Visual timer terminates
        self.mock_sd.stop.assert_called_once()                         # Hardware drivers silenced
        
        # Trigger Ctrl+Right Click (Resume Audio Mode manually at +0.5 offset)
        self.app.paused_x = 0.5
        self.app.toggle_pause()
        self.assertFalse(self.app.paused)                              # Flag unlocks exactly
        self.assertTrue(self.app.playback_timer.isActive())            # Animation cache fires
        self.mock_sd.play.assert_called()                              # Sounddevice picks up exactly at frame slice

        self.mock_sd.play.assert_called()                              # Sounddevice picks up exactly at frame slice
        
    @patch("AnnotatorApp.AnnotationEditDialog")
    def test_double_click_focus(self, mock_dlg_class):
        # Create an annotation at [0.5s - 0.6s]
        mock_dlg = MagicMock()
        mock_dlg.exec_.return_value = 1 
        mock_dlg.class_label = "test-focus"
        mock_dlg.weight = 0.0
        mock_dlg_class.return_value = mock_dlg
        
        self.app.on_select_interval(0.5, 0.6)
        item = self.app.tree.topLevelItem(0)
        
        # Set window duration to 0.2s for easy verification
        self.app.entry_duration.setText("0.2")
        
        # Trigger double click
        self.app.on_item_double_clicked(item, 0)
        
        # Expected new center: 0.55s. duration: 0.2s -> range: [0.45, 0.65]
        xlim = self.app.ax.get_xlim()
        self.assertAlmostEqual(xlim[0], 0.45, places=2)
        self.assertAlmostEqual(xlim[1], 0.65, places=2)
        self.assertEqual(self.app.entry_start_time.text(), "0.450")

if __name__ == '__main__':
    unittest.main()
