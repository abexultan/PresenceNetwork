from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch
from torchvision import transforms
import pickle
import cv2 as cv
from skimage.util import random_noise

mean = (.5, .5, .5)
std = (.5, .5, .5)
normalize = transforms.Normalize(mean, std)


def transform(inp, shape, normalize=False):
    
    if type(inp) == np.ndarray:
        inp_tensor = torch.from_numpy(inp)
        inp_tensor = inp_tensor.type(torch.FloatTensor)
    else:
        inp_tensor = torch.tensor(inp)
    
    inp_tensor = inp_tensor.view(shape)
    
    if normalize:
        inp_tensor = normalize(inp_tensor)
    
    return inp_tensor

class PresenceDataset(Dataset):
    
    augment_dict = {1: "rotate", 2: "shift"}

    def __init__(self, data_path, annotation_path, norm=normalize):

        super().__init__()
        self.coco = COCO(annotation_path)
        with open('./present_list.txt', 'rb') as fb:
            self.present_list = pickle.load(fb)
        with open('./non_present_list.txt', 'rb') as fb:
            self.non_present_list = pickle.load(fb)
        self.data_list = self.present_list + self.non_present_list
        self.cats = set(self.coco.getCatIds())
        self.data_path = data_path
        self.normalize = normalize
        # getting the list of IDs associated with present GT pair

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, idx):

        img_id = self.data_list[idx]
        ann_lst = self.coco.getAnnIds(img_id, iscrowd=False, areaRng=[10000, float('inf')])

        if ann_lst is []:
            return None

        else:

            present_set = set()
            for annot_id in ann_lst:
                cat = self.coco.loadAnns(annot_id)[0]['category_id']
                if cat not in present_set:
                    present_set.add(cat)
                else:
                    continue

            # getting the target depending on presence of GT pair
            if img_id in self.present_list:
                target, presence, _, _ = self._crop_random(present_set,
                                                           img_id, True, idx)
            else:
                non_present_lst = list(self.cats - present_set)
                target, presence, _, _ = self._crop_random(non_present_lst,
                                                           img_id, False, idx)

            img_meta = self.coco.loadImgs(img_id)[0]
            img = np.array(Image.open(''.join([self.data_path, '/',
                                               img_meta['file_name']])))

            img = self.__class__._rescale(img, (512, 512), self.normalize)
            target = self.__class__._rescale(target, (96, 96), self.normalize)

            if presence:
                out = transform([1,0], (1,2), False)
            else:
                out = transform([0,1], (1,2), False)

            return {"image_pair": (img, target), "presence": out}

    @staticmethod
    def _rescale(img, shape, normalize):

        img_obj = Image.fromarray(img)
        img_obj.thumbnail(size=shape)
        resized_img = Image.new('RGB', shape, 0)
        resized_img.paste(img_obj, ((shape[0] - img_obj.size[0])//2,
                                    (shape[1] - img_obj.size[1])//2))
        resized_img = np.array(resized_img)
        
        return transform(resized_img,
                         (3, shape[0], shape[1]),
                         normalize)

    def _crop_random(self, category_lst, img_id, is_present, idx):
        '''
        Crops the target image and returns both target array
        and presence boolean

        Args: category_list - used to get the random image where
                            certain category is present

              img_id - ID of the scene from pair, used if is_present
                       boolean is True to assure the cropped
                       target being from another image

              is_present - if True, will remove ID of the image
                           from possible candidate for target list

        Output: cropped - proposed target cropped image

                is_present - presence of target in the img_id
        '''

        if is_present:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, areaRng=[64 ** 2, 96 ** 2], iscrowd=False)
            
        
        else: 
            ann_ids = self.coco.getAnnIds(catIds=category_lst,
                                          areaRng=[64**2, 96**2], iscrowd=False)
        
        annotation = self.coco.loadAnns(random.choice(ann_ids))[0]
        target_id = annotation['image_id']

        # bbox of target
        x, y, w, h = annotation['bbox']
        area = annotation['area']

        # cropping the target with specified bounding box
        meta = self.coco.loadImgs(target_id)[0]
        arr_img = np.array(Image.open(''.join([self.data_path, '/',
                                               meta['file_name']])))
        cropped = arr_img[int(y):int(y+h), int(x):int(x+w)]
        
        if is_present:
            augment_id = random.randint(1,2)
            augment = self.augment_dict[augment_id]
            if augment == "rotate":
                cropped = cv.rotate(cropped, cv.ROTATE_90_CLOCKWISE)
            elif augment == "shift":
                cropped = arr_img[int(y)+20:int(y+h)+20, int(x):int(x+w)+20]
            
            # adding salt and pepper noize
            cropped = random_noise(cropped)
                

        return cropped, is_present, area, annotation['category_id']
