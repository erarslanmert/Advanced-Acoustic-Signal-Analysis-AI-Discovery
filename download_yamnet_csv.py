import urllib.request
url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
urllib.request.urlretrieve(url, 'yamnet_class_map.csv')
print("Downloaded yamnet_class_map.csv")
