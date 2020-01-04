import cv2
import urllib.request
import json
from pandas.io.json import json_normalize
import logging
from matplotlib import pyplot as plt
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import datetime
from pylab import rcParams
from sklearn.cluster import KMeans
import numpy as np
import os


class VisualRecognition:
    def __init__(self, image_urls=None, user_id=None, compress_images: bool = True):
        self.name = self.__class__.__name__

        # initialize logfile
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        root_log_path = './log'
        log_file = '/{}_{}.log'.format(self.name, timestamp)
        if not os.path.isdir(root_log_path):
            os.mkdir(root_log_path)
        # initialize logger and std out handler
        logging.basicConfig(filename=root_log_path+log_file, level=logging.DEBUG)
        if logging.StreamHandler not in [type(handler) for handler in logging.getLogger().handlers]:
            logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Logging Started: {}".format(log_file))

        # define image urls
        if image_urls is None:
            self.urls = ['https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Hotdog_-_Evan_Swigart.jpg/220px-Hotdog_-_Evan_Swigart.jpg',
                         'https://i5.walmartimages.ca/images/Enlarge/094/510/6000200094510.jpg']
        if type(image_urls) is str:
            self.urls = [image_urls]

        # init other attributes
        self.last_response = None

        # get api key from userdata.json
        if user_id is not None:
            rd_tkn = self.get_token(user_id)
            self.api_key = rd_tkn

        # get/show images
        logging.info("Collecting Images...")
        self.imgs = [self.get_image(url) for url in self.urls]
        if compress_images:
            depth = 16  # TODO: make a var
            self.kmeans = KMeans(n_clusters=depth)
            self.imgs = [self.compress_image(img) for img in self.imgs]

        # call watson
        logging.info("Calling Watson...")
        authenticator = IAMAuthenticator(self.api_key)
        self.watson = VisualRecognitionV3('2018-03-19', authenticator=authenticator)

        # classify images
        logging.info("Classifying Images...")
        self.dfs = []
        for url in self.urls:
            classes = self.watson.classify(url=url,
                                           threshold=0.5,
                                           classifier_ids=['default']).get_result()
            self.dfs.append(self.watson_to_df(classes))

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.info("{} Destroyed at {}.".format(self.name, datetime.datetime.now()))

    @staticmethod
    def get_token(user_id):
        with open('./db/udata.json') as json_data:
            data = json.load(json_data)[user_id]
        return [token['api_key'] for token in data['tokens'] if token['site'] == 'ibm'].pop()

    @staticmethod
    def watson_to_df(json_response):
        json_classes = json_response['images'][0]['classifiers'][0]['classes']
        df = json_normalize(json_classes).sort_values('score', ascending=False).reset_index(drop=True)
        return df

    @staticmethod
    def show_image(img, size=(800, 600), window_name='Press Any Key To Continue.'):
        rcParams['figure.figsize'] = size[0], size[1]
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, size[0], size[1])
        cv2.imshow(window_name, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def get_image(self, img_url, greyscale: bool = False):
        self.last_response = urllib.request.urlretrieve(img_url)
        image = cv2.imread(self.last_response[0])

        if greyscale:
            color_type = cv2.COLOR_BGR2GRAY
            image = cv2.cvtColor(image, color_type)
        else:
            color_type = cv2.COLOR_BGR2RGB

        return image

    def compress_image(self, image):
        logging.info("Compressing Image...")
        im_width = image.shape[0]
        im_height = image.shape[1]
        image_reshaped = image.reshape(im_width * im_height, 3)
        self.kmeans.fit(image_reshaped)
        cluster_locs = np.asarray(self.kmeans.cluster_centers_, dtype=np.uint8)
        labels = np.asarray(self.kmeans.labels_, dtype=np.uint8)
        labels = labels.reshape(im_width, im_height)

        compressed_image = np.ones((im_width, im_height, 3), dtype=np.uint8)
        for r in range(im_width):
            for c in range(im_height):
                compressed_image[r, c, :] = cluster_locs[labels[r, c], :]
        logging.info("Image Compressed.")
        return compressed_image

    def canny_edges(self, image):
        edges = cv2.Canny(image,
                          threshold1=100,
                          threshold2=200)
        return edges

    def image_hist(self, image, window_name='Press Any Key To Continue.'):
        rcParams['figure.figsize'] = 8, 4

        # find color depth, then hist each color after cv2 color transform
        if image.shape[2] == 3:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histr = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
            plt.show()
        else:
            plt.hist(image.ravel(), 256, [0, 256])
            plt.title(window_name)
            plt.show()


if __name__ == '__main__':
    """
        -- my encrypted user id from udata.json, api_key is not exposed --
        -- below is udata.json format, if you wanna replicate it put in ./db/udata.json --
    {
        '<user_id_here>': {
            "firstName": <your_name_here>,
            "tokens": [
                {
                    "site": "ibm",
                    "site_url": '<api_url_here>',
                    "api_key": '<api_key_here>'
                }
            ]
    }
    """

    vr = VisualRecognition(user_id="6541764432777451972", compress_images=False)
    [vr.show_image(img) for img in vr.imgs]
    [vr.show_image(vr.canny_edges(img)) for img in vr.imgs]
    [vr.image_hist(img) for img in vr.imgs]
    [print(df) for df in vr.dfs]
    logging.info("Done")