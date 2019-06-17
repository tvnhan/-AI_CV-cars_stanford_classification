import os
import csv
import sys
sys.path.append('../')

from torch.autograd import Variable
import torch.utils.data
from torch.nn import DataParallel
import main_source.models.NTS.model as model
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torch.autograd import Variable


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def _load_name_cars(link_csv):
    list_data = []
    with open(link_csv) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter='|')
        for row in csvReader:
            list_data.append(row)
    return list_data

def _convert_code_to_name(list_data, code):
    return list_data[code]

def _load_model(model_path):
    net = model.attention_net(topN=5)
    ckpt = torch.load(model_path)
    net.load_state_dict(ckpt['net_state_dict'])
    net = net.cuda()
    net = DataParallel(net)
    return net


def predict_img_link(net, img_link):
    net.eval()
    with torch.no_grad():
        image = Image.open(img_link).convert('RGB')  # (C, H, W)
        transform = transforms.Compose([
            transforms.Resize(size=(448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)

        image = image.to('cuda:0')
        image = image.view(1, *image.size())
        image = Variable(image)

        _, concat_logits, _, _, _ = net(image)
        _, concat_predict = torch.max(concat_logits, 1)
        print(concat_predict.data.item())
        return concat_predict.data.item()


if __name__ == '__main__':
    model_path = '../outputs/best_model.ckpt'
    net = _load_model(model_path)

    link_car_name = '../dataset/names.csv'
    list_data = _load_name_cars(link_car_name)

    img_link = '../dataset/val/00003.jpg'
    result = predict_img_link(net, img_link)
    print(_convert_code_to_name(list_data, int(result)))



