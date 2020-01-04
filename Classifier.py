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
import os


class VisualRecognition:
    def __init__(self, image_urls=None, user_id=None, refresh: bool = True):
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
            self.urls = ['http://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/CV0101/Images/Donald_Trump_Justin_Trudeau_2017-02-13_02.jpg', 'https://i5.walmartimages.ca/images/Enlarge/094/510/6000200094510.jpg']
        if type(image_urls) is str:
            self.urls = [image_urls]

        # init other attributes
        self.refresh = refresh
        self.last_response = None

        # get api key from userdata.json
        if user_id is not None:
            rd_tkn = self.get_token(user_id)
            self.api_key = rd_tkn

        # get/show images
        self.imgs = [self.get_image(url) for url in self.urls]
        # [self.show_image(img) for img in self.imgs]

        # call watson
        authenticator = IAMAuthenticator(self.api_key)
        self.watson = VisualRecognitionV3('2018-03-19', authenticator=authenticator)

        # classify images
        self.dfs = []
        for url in self.urls:
            classes = self.watson.classify(url=url,
                                           threshold=0.5,
                                           classifier_ids=['default']).get_result()
            # dump = json.dumps(classes, indent=2)
            # print(dump)
            self.dfs.append(self.watson_to_df(classes))

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

    def get_image(self, img_url):
        self.last_response = urllib.request.urlretrieve(img_url)
        image = cv2.imread(self.last_response[0])
        return image


if __name__ == '__main__':
    # my encrypted user id from udata.json, api_key is not exposed
    # udata.json format, if you wanna replicate it put in ./db/udata.json
    # {
    #     '<user_id_here>': {
    #         "firstName": <your_name_here>,
    #         "tokens": [
    #             {
    #                 "site": "ibm",
    #                 "site_url": '<api_url_here>',
    #                 "api_key": '<api_key_here>'
    #             }
    #         ]
    # }
    vr = VisualRecognition(user_id="6541764432777451972")
