import torch.optim as optim
import torch.backends.cudnn as cudnn
from time import time
import torch.nn as nn
from dataset import *
from models import *
from distill import *
from utils import *
from argparse import Namespace


args = Namespace()
args.data_dir      = "/content/gdrive/MyDrive/"
args.data          = 'CIFAR100'
args.epoch         = 200
args.batch_size    = 64
args.lr            = 0.05
args.momentum      = 0.9
args.lr_decay      = 0.1
args.weight_decay  = 5e-4
args.alpha         = 0.9
args.temperature   = 4
args.model_t       = 'wrn40x2'
args.model         = 'resnet56'
args.beta          = 200
args.qk_dim        = 128
args.trained_dir   = "/content/wrn40x2.pth"
args.schedule      = [150, 180, 210]


def test_accuracy(model, dataloader, device="cuda"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
      for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def main():
    args.guide_layers = LAYER[args.model_t]
    args.hint_layers  = LAYER[args.model]

    for param in sorted(vars(args).keys()):
        print('--{0} {1}'.format(param, vars(args)[param]))

    train_loader, test_loader, args.num_classes, args.image_size = create_loader(args.batch_size, args.data_dir, args.data)

    model_t = wrn40x2(num_classes= args.num_classes)
    model_t.load_state_dict(torch.load(args.trained_dir ) )

    model_s = wrn16x2(num_classes=args.num_classes)

    device = torch.device('cuda')
    
    #1
    data = torch.randn(2, 3, args.image_size, args.image_size)

    model_t.eval()
    model_s.eval()

    # list of intermediate tensors
    with torch.no_grad():
        feat_t, _ = model_t(data, is_feat=True)
        feat_s, _ = model_s(data, is_feat=True)

    # all components
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    
    # student + AFD
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    # needed for down-sampling and up-sampling
    args.s_shapes = [feat_s[i].size() for i in args.hint_layers]
    args.t_shapes = [feat_t[i].size() for i in args.guide_layers]
    args.n_t, args.unique_t_shapes = unique_shape(args.t_shapes)


    criterion_ce = nn.CrossEntropyLoss()
    # KL-divergence
    criterion_kl = DistillKL(args.temperature) # not used.
    criterion_kd = AFD(args)
    
    module_list.append(criterion_kd)
    trainable_list.append(criterion_kd)

    criterion = nn.ModuleList([])
    criterion.append(criterion_ce)
    criterion.append(criterion_kl)
    criterion.append(criterion_kd)

    optimizer = optim.SGD(trainable_list.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    module_list.append(model_t)
    module_list.cuda()
    criterion.cuda()
    cudnn.benchmark = True
    best_last_acc = 0
    Save_file = "s_model.pth"

    for epoch in range(1, args.epoch + 1):
        s = time()
        adjust_learning_rate(optimizer, epoch, args) # /=10 learning rate

        train_loss, train_acc1, train_acc5 = train_kl(module_list, optimizer, criterion, train_loader, device, args)
        test_acc1, test_acc5 = test(model_s, test_loader, device)
        if best_last_acc < test_acc1:
            torch.save(model_s.state_dict(), Save_file)
            best_last_acc = test_acc1

        print('Epoch: {0:>3d} |Train Loss: {1:>2.4f} |Train Top1: {2:.4f} |Train Top5: {3:.4f} |Test Top1: {4:.4f} |Test Top5: {5:.4f}| Time: {6:>5.1f} (s)'
            .format(epoch, train_loss, train_acc1, train_acc5, test_acc1, test_acc5, time() - s))

    acc_t = test_accuracy(model_t, test_loader)
    print("Teacher Accuracy on test data:",acc_t)
    acc_s = test_accuracy(model_s, test_loader)
    print("Student Accuracy on test data:",acc_t)

if __name__ == "__main__":
    main()