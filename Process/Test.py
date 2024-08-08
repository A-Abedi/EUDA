import torch
from Utils.transform import get_transform

from Utils.data_utils import ImageList
from .Valid import valid


def test(args, model):
    model.load_state_dict(torch.load(args.pretrained_dir))
    model.to(args.device)

    _, transform_test = get_transform(args.img_size)
    test_loader = torch.utils.data.DataLoader(
        ImageList(open(args.test_list).readlines(), args.dataset_path, transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    accuracy, classWise_acc = valid(args, model, test_loader, global_step=1)
    print(accuracy)
    print(classWise_acc)
