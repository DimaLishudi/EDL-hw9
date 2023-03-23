import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs


class LayerOutputResNet(nn.Module):
    """
    Wrapper around ResNet101 to get layer output for knowledge distillation
    """
    def __init__(self, inner_model):
        super().__init__()
        self.model = inner_model
        self.preproc_layers = ["conv1", "bn1", "relu", "maxpool", "layer1"]

    def forward(self, input):
        return self.model(input)

    def layer_output(self, input):
        out_1 = input
        for layer in self.preproc_layers:
            out_1 = getattr(self.model, layer)(out_1)
        out_2 = self.model.layer2(out_1)
        out_4 = self.model.layer4(self.model.layer3(out_2))
        out = self.model.fc(self.model.avgpool(out_4).flatten(1))
        return out_1, out_2, out_4, out



def get_student():
    model = LayerOutputResNet(torchvision.models.resnet101())
    model.model.fc = nn.Linear(2048, 10, bias=True)
    model.layer3 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    return model



@torch.inference_mode()
def evaluate(model, loader, device):
    correct = 0
    total = 0
    model.eval()
    model.to(device)
    for img, label in loader:
        total += label.shape[0]
        img = img.to(device)
        out = model(img)
        correct += torch.sum(out.argmax(dim=-1).cpu() == label).item()
    return correct/total


def train(model, teacher, train_epoch_fn, train_loader, test_loader, device, title):
    print("\n" + title, "="*80 + "\n")
    optimizer = torch.optim.Adam(model.parameters())
    stop_counter = 0
    accs = []
    epoch = 0 
    
    while True:
        train_epoch_fn(model, teacher, optimizer, train_loader, device)
        accs.append(evaluate(model, test_loader, device))
        print(f"Epoch {epoch} acc: {accs[-1]}")

        if epoch > 0 and np.abs(accs[-1] - accs[-2]) < 0.01:
            stop_counter += 1
        else:
            stop_counter = 0
        if stop_counter == 2:
            break
        epoch += 1
    return accs


def train_epoch_simple(model, _, optimizer, loader, device):
    model.train()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    for img, label in loader:
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()



def train_epoch_out_distill(model, teacher, optimizer, loader, device):
    model.train()
    teacher.eval()
    model.to(device)
    teacher.to(device)
    loss_fn = nn.CrossEntropyLoss()
    for img, label in loader:
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        st_out = model(img)
        with torch.no_grad():
            t_out = teacher(img).softmax(dim=-1)
        loss = (loss_fn(st_out, label) + loss_fn(st_out, t_out)) / 2
        loss.backward()
        optimizer.step()


def train_epoch_layer_distill(model, teacher, optimizer, loader, device):
    model.train()
    teacher.eval()
    model.to(device)
    teacher.to(device)
    label_loss_fn = nn.CrossEntropyLoss()
    layer_loss_fn = nn.MSELoss()
    for img, label in loader:
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        st_1, st_2, st_4, st_out = model.layer_output(img)
        with torch.no_grad():
            t_1, t_2, t_4, t_out = teacher.layer_output(img)
            t_out = t_out.softmax(dim=-1)
        loss = (label_loss_fn(st_out, label) + \
                label_loss_fn(st_out, t_out) + \
                layer_loss_fn(st_1, t_1) + \
                layer_loss_fn(st_2, t_2) + \
                layer_loss_fn(st_4, t_4)) / 5
        loss.backward()
        optimizer.step()


def main():
    # set-up
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
    teacher = LayerOutputResNet(torchvision.models.resnet101(weights=weights, progress=False))
    teacher.model.fc = nn.Linear(2048, 10, bias=True)

    train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, transform=weights.transforms(), download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, transform=weights.transforms(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, pin_memory=True, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=12, pin_memory=True, shuffle=False, drop_last=False)

    makedirs("models", exist_ok=True)

    # part 0 ==================================================================
    
    teacher_accs = train(teacher, None, train_epoch_simple, train_loader, test_loader, device, title="ResNet101 Finetune")
    torch.save(teacher.model.state_dict(), "models/teacher.pth")

    # part 1 ==================================================================
    
    student = get_student()
    student_accs_1 = train(student, None, train_epoch_simple, train_loader, test_loader, device, title="Data Only Student")
    torch.save(student.model.state_dict(), "models/student1.pth")
    del student

    # part 2 ==================================================================
    
    student = get_student()
    student_accs_2 = train(student, teacher, train_epoch_out_distill, train_loader, test_loader, device, title="Data + Teacher labels Student")
    torch.save(student.model.state_dict(), "models/student2.pth")
    del student

    # part 3 ==================================================================
    
    student = get_student()
    student_accs_3 = train(student, teacher, train_epoch_layer_distill, train_loader, test_loader, device, title="Data + Teacher labels + Teacher layers Student")
    torch.save(student.model.state_dict(), "models/student3.pth")
    del student

    # plot results
    plt.figure(figsize=(12,8))
    plt.plot(teacher_accs, label="ResNet101 Finetune")
    plt.plot(student_accs_1, label="Data Only Student")
    plt.plot(student_accs_2, label="Data + Teacher labels Student")
    plt.plot(student_accs_3, label="Data + Teacher labels + Teacher layers Student")
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.title("Test Accuracy During Training for Different Setups")
    plt.legend()
    plt.savefig("task1.pdf")
    plt.show()

if __name__ == "__main__":
    main()