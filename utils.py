from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d


class DataExtractor():
    """Class for extracting data from dataset, such as masks of segmentation and pose coordinates."""
    def __init__(self):
        """
        Attributes:
            garment_colors (dict): Contains lists of RGB codes of segmented parts.
        """
        self.garment_colors = {
            'top': [(64, 128, 128), (192, 128, 128), (128, 64, 128), 
                    (128, 128, 128), (128, 0, 128)],
            'bottom': [(64, 0, 128), (192, 0, 0)]
        }
    
    def get_pose_coordinates(self, file_num):
        """Obtains coordinates (x, y) from /dataset/pose_json for given file number.
        Args:
            file_num (str): Number of file in dataset
        """
        with open(f'./dataset/pose_json/{file_num}_00_keypoints.json') as f:
            contents = json.load(f) 
        data = contents['people'][0]['pose_keypoints_2d']
        # data has format [x, y, p, x, y, p,...]
        x = data[0::3]
        y = data[1::3]
        return x, y
    
    def extract_masks_from_segmentation(self, file_num):
        """Extracts segmentation masks for top and bottom from /dataset/human_parsing.
        Args:
            file_num (str): number of file in dataset
        Returns:
            mask_top: Mask of top of the body (blouse, hands, ets.) 
            mask_bottom: Mask of lower of the body (skirt)
        """
        path = f'./dataset/human_parsing/{file_num}_00.png'
        im = Image.open(path).convert('RGB')
        im_pix = np.array(im)
        
        mask_top = np.full((1024, 768), False)
        for color in self.garment_colors['top']:
            mask = (im_pix == color).all(-1)
            mask_top = mask_top | mask
            
        mask_bottom = np.full((1024, 768), False)
        for color in self.garment_colors['bottom']:
            mask = (im_pix == color).all(-1)
            mask_bottom = mask_bottom | mask
        return mask_top, mask_bottom
    
    
class MaskGenerator():
    """Class for obtaining mask of big skirt"""
    def __init__(self, data_extr):
        """
        Args:
            data_extr (DataExtractor): Object of class Data for interacting with dataset
            custom_pose_dict (dict): Coordinates of 6 boundary points of big skirt 
                from first 19 imgs that are needed for linear regression.
            pose_dict (dict): Coordinates of 6 points (2 hips, 2 knees, 2 ankles), 
                first 19 imgs, that are needed for linear regression.
            linreg_1: Linear Regression object that uses coordinates of hips
                to predicts 2 top points of big skirt
            linreg_2: Linear Regression object that uses coordinates of knees
                to predicts 2 middle points of big skirt
            linreg_3: Linear Regression object that uses coordinates of ankles
                to predicts 2 bottom points of big skirt
        """
        self.data_extr = data_extr
        self.custom_pose_dict = self.__get_custom_pose_dict()
        self.pose_dict = self.__get_pose_dict()
        
        self.linreg_1 = LinearRegression()
        self.linreg_2 = LinearRegression()
        self.linreg_3 = LinearRegression()
        self.__fit_linregs()
    
    def __get_custom_pose_dict(self):
        """Obtains coordinates of 6 boundary points of big skirt from first 19 imgs.
        This coordinates are needed for linear regression. Obtained manually using guideline."""
        with open('./custom_pose.json', 'r') as f:
            custom_pose_dict = json.load(f)
        return custom_pose_dict
    
    def __get_pose_dict(self):
        """Obtains coordinates of 6 points (2 hips, 2 knees, 2 ankles) of first 19 imgs from dataset.
        This coordinates are needed for linear regression."""
        pose_dict = {}
        for file_num in self.custom_pose_dict.keys():
            x, y = self.data_extr.get_pose_coordinates(file_num)
            # hips coords
            pose_dict[file_num] = [x[9], y[9], x[12], y[12]] 
            # knees coords
            pose_dict[file_num] += [x[10], y[10], x[13], y[13]]
            # anckles coords
            pose_dict[file_num] += [x[11], y[11], x[14], y[14]]
        return pose_dict
    
    def __fit_linregs(self):
        """Fits 3 objects of Linear Regression that are used to predict 6 boundary points of big skirt.
        linreg_1 predicts 2 top points of big skirt. linreg_2 predicts 2 middle points of big skirt.
        linreg_3 predicts 2 bottom points of big skirt."""
        pose_arr = np.array(list(self.pose_dict.values()))
        custom_pose_arr = np.array(list(self.custom_pose_dict.values()))
        self.linreg_1.fit(pose_arr[:,:4], custom_pose_arr[:,:4])
        self.linreg_2.fit(pose_arr[:,4:8], custom_pose_arr[:,4:8])
        self.linreg_3.fit(pose_arr[:,8:], custom_pose_arr[:,8:])
        
    def _predict_points(self, file_num):
        """Predicts 6 boundary points of big skirt. linreg_1 predicts 2 top points of big skirt.
        linreg_2 predicts 2 middle points of big skirt. linreg_3 predicts 2 bottom points of big skirt."""
        x, y = self.data_extr.get_pose_coordinates(file_num)
        # Case: points are swaped
        # hips are swaped
        if x[9] > x[12]:
            x[9], x[12] = x[12], x[9]
            y[9], y[12] = y[12], y[9]
        # knees are swaped
        if x[10] > x[13]:
            x[10], x[13] = x[13], x[10]
            y[10], y[13] = y[13], y[10]
        # ankles are swaped
        if x[11] > x[14]:
            x[11], x[14] = x[14], x[11]
            y[11], y[14] = y[14], y[11]
        
        # Prediction
        pred_1 = self.linreg_1.predict(np.array([[x[9], y[9], x[12], y[12]]])).reshape(2, 2)
        pred_2 = self.linreg_2.predict(np.array([[x[10], y[10], x[13], y[13]]])).reshape(2, 2)
        pred_3 = self.linreg_3.predict(np.array([[x[11], y[11], x[14], y[14]]])).reshape(2, 2)
        
        # Case: no ankles    
        if (x[11], y[11], x[14], y[14]) == (0, 0, 0, 0):
            pred_3 = pred_2.copy()
        res = np.vstack([pred_1, pred_2, pred_3])
        return res
    
    def _calc_skirt_points(self, file_num):
        """Calculates interpolated skirt points using 6 predicted points"""
        # Obtaining 6 skirt points and reordering them
        points = self._predict_points(file_num)
        order = [0, 2, 4, 5, 3, 1]
        points = points[order]
        x, y = points[:,0], points[:, 1]
        
        # Adding ankle points
        x_pose, y_pose = self.data_extr.get_pose_coordinates(file_num)
        # Case: ankles are swaped
        if x_pose[11] > x_pose[14]:
            x_pose[11], x_pose[14] = x_pose[14], x_pose[11]
            y_pose[11], y_pose[14] = y_pose[14], y_pose[11]
        # Case: no ankles    
        if (x_pose[11], y_pose[11], x_pose[14], y_pose[14]) == (0, 0, 0, 0):
            x_pose[11], x_pose[14] = x[2], x[2]
            y_pose[11], y_pose[14] = y[2], y[2]
        x = np.insert(x, [3, 3], [x_pose[11], x_pose[14]])
        y = np.insert(y, [3, 3], [y_pose[11], y_pose[14]])
        
        # Interpolate 3 segments (quadratic spline)
        x_res, y_res = np.array([]), np.array([])
        segments= [[0, 1, 2], [2, 3, 4, 5], [5, 6, 7]]
        for segment in segments:
            t = np.arange(len(segment))
            ti = np.linspace(0, t.max(), 2 * t.size)
            xi = interp1d(t, x[segment], kind='quadratic')(ti)
            yi = interp1d(t, y[segment], kind='quadratic')(ti)
            x_res = np.append(x_res, xi[:-1], 0)
            y_res = np.append(y_res, yi[:-1], 0)
        # Adding last point
        x_res = np.append(x_res, xi[[-1]], 0)
        y_res = np.append(y_res, yi[[-1]], 0)
        points = np.column_stack((x_res, y_res))
        return points
    
    def _calc_skirt_mask(self, file_num):
        """Obtains mask of big skirt using interpolated skirt points"""
        points = self._calc_skirt_points(file_num)
        img = Image.new(mode='1', size=(768, 1024))
        ImageDraw.Draw(img).polygon(list(points.reshape(-1)), outline=1, fill=1)
        mask_skirt = np.array(img, dtype=bool)
        return mask_skirt
    
    def calc_result_mask(self, file_num):
        """Obtains mask of big skirt that corrected with segmentation masks"""
        mask_top, mask_bottom = self.data_extr.extract_masks_from_segmentation(file_num)
        mask_skirt = self._calc_skirt_mask(file_num)
        mask_result = ~mask_top & mask_skirt | mask_bottom
        return mask_result