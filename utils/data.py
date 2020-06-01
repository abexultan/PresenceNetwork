from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random


class PresenceDataset(Dataset):

    def __init__(self, data_path, annotation_path, transform=None):

        super().__init__()
        self.coco = COCO(annotation_path)
        self.data_list = self.coco.getImgIds()
        self.cat_list = self.coco.getCatIds()
        self.data_path = data_path
        if not transform:
            transform = []
        self.transform = transform
        # getting the list of IDs associated with present GT pair
        self.present = self.data_list[:len(self.data_list)//2]


    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, idx):

        check = False
        while not check:
            img_id = self.data_list[idx]
            ann_lst = self.coco.getAnnIds(img_id)
            if ann_lst:
                check = True

        present_list = []
        for annot_id in ann_lst:
            cat = self.coco.loadAnns(annot_id)[0]['category_id']
            is_crowd = bool(self.coco.loadAnns(annot_id)[0]['iscrowd'])
            if cat not in present_list and not is_crowd:
                present_list.append(cat)
            else:
                continue
#         print("ID of the image : {}".format(img_id))
        # getting the target depending on presence of GT pair
        if img_id in self.present:
            target, presence, _, _ = self._crop_random(present_list,
                                                 img_id, True, idx)
        else:
            non_present_lst = list(set(self.cat_list) - set(present_list))
            target, presence, _, _ = self._crop_random(non_present_lst,
                                                 img_id, False, idx)

        img_meta = self.coco.loadImgs(img_id)[0]
        img = np.array(Image.open(''.join([self.data_path, '/',
                                           img_meta['file_name']])))

        img = self.__class__._rescale(img, [512, 512])
        target = self.__class__._rescale(target, [96, 96])

        if presence:
            out = [1, 0]
        else:
            out = [0, 1]

        return {"image_pair": (img, target), "presence" : out}

    @staticmethod
    def _rescale(img, shape):

        img_obj = Image.fromarray(img)
        img_obj.thumbnail(size=shape)
        resized_img = Image.new('RGB', shape, 0)
        resized_img.paste(img_obj, ((shape[0] - img_obj.size[0])//2,
                                        (shape[1] - img_obj.size[1])//2))
        return np.array(resized_img)


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

        # get the list of images assosciated with category
        target_lst = self.coco.getImgIds(catIds=category_lst)

        if is_present:
            target_lst = set(target_lst)
            target_lst.remove(img_id)
            target_lst = list(target_lst)

        # get the random image ID and category's bounding box
        ann_ids = self.coco.getAnnIds(catIds=category_lst, imgIds=target_lst, areaRng=[10000, float('inf')])
        # handling the case of image and target being the same crop
        annotation = self.coco.loadAnns(random.choice(ann_ids))[0]
        target_id = annotation['image_id']

        # bbox of target
        x, y, w, h = annotation['bbox']
        area = annotation['area']

        # cropping the target with specified bounding box
        meta = self.coco.loadImgs(target_id)[0]
        arr_img = np.array(Image.open(''.join([self.data_path, '/', meta['file_name']])))
        cropped = arr_img[int(y):int(y+h), int(x):int(x+w)]

        return cropped, is_present, area, annotation['category_id']
