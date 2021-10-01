from PIL import Image
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from evaluation.recall import recall_at_ks
import time
cudnn.benchmark = True

import net


def extractDeepFeature(img, model, is_gray):
    if is_gray:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5,), std=(0.5,))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    img, img_ = transform(img), transform(F.hflip(img))
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    ft = torch.cat((model(img), model(img_)), 1)[0].to('cpu')
    return ft


def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i * n / n_folds:(i + 1) * n / n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def eval(model, model_path=None, is_gray=False):
    predicts = []
    model.load_state_dict(torch.load(model_path))
    model.eval()
    root = '/home/wangyf/dataset/lfw/lfw-112X96/'
    with open('/home/wangyf/Project/sphereface/test/data/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]

    with torch.no_grad():
        for i in range(6000):
            p = pairs_lines[i].replace('\n', '').split('\t')

            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")

            with open(root + name1, 'rb') as f:
                img1 =  Image.open(f).convert('RGB')
            with open(root + name2, 'rb') as f:
                img2 =  Image.open(f).convert('RGB')
            f1 = extractDeepFeature(img1, model, is_gray)
            f2 = extractDeepFeature(img2, model, is_gray)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, distance, sameflag))

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts = np.array(map(lambda line: line.strip('\n').split(), predicts))
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

    return np.mean(accuracy), predicts


def predict_batchwise(model, dataloader):
    '''
        Predict on a batch
        :return: list with N lists, where N = |{image, label, index}|
    '''
    # print(list(model.parameters())[0].device)
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc="Batch-wise prediction"):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)
                    # predict model output for image
                    J = model(J).cpu()
                for j in J:
                    #if i == 1: print(j)
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    return [torch.stack(A[i]) for i in range(len(A))]

def evaluate(model, dataloader, eval_nmi=False, recall_list=[1, 2, 4, 8]):
    '''
        Evaluation on dataloader
        :param model: embedding model
        :param dataloader: dataloader
        :param eval_nmi: evaluate NMI (Mutual information between clustering on embedding and the gt class labels) or not
        :param recall_list: recall@K
    '''
    eval_time = time.time()
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, *_ = predict_batchwise(model, dataloader)
    print('done collecting prediction')

    nmi, recall = recall_at_ks(X, T, ks=recall_list)
    for i in recall_list:
        print("Recall@{} {:.3f}".format(i, recall[i]))
    return nmi, recall



if __name__ == '__main__':
    _, result = eval(net.sphere().to('cuda'), model_path='checkpoint/CosFace_24_checkpoint.pth')
    np.savetxt("result.txt", result, '%s')
