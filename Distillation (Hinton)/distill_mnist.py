import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

class MnistTeacher(nn.Module):
    def __init__(self):
        super(MnistTeacher, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 3 * 3, 625)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(625, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

class MnistStudent(nn.Module):
    def __init__(self):
        super(MnistStudent, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_mnist_local(data_dir):
    def read_idx3_ubyte(file_path):
        with open(file_path, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        return data

    def read_idx1_ubyte(file_path):
        with open(file_path, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            num_items = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

    train_images = read_idx3_ubyte(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    train_labels = read_idx1_ubyte(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    test_images = read_idx3_ubyte(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    test_labels = read_idx1_ubyte(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))

    return train_images, train_labels, test_images, test_labels

data_dir = 'your dataset link'
train_images, train_labels, test_images, test_labels = load_mnist_local(data_dir)

train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1) / 255.0
test_labels = torch.tensor(test_labels, dtype=torch.long)

train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

teacher = MnistTeacher()
student = MnistStudent()

optimizer_teacher = optim.RMSprop(teacher.parameters(), lr=1e-4)
optimizer_student = optim.Adam(student.parameters(), lr=1e-3)

def distillation_loss(student_output, teacher_output, temperature):
    soft_teacher = torch.softmax(teacher_output / temperature, dim=1)
    soft_student = torch.softmax(student_output / temperature, dim=1)
    return torch.mean(-torch.sum(soft_teacher * torch.log(soft_student), dim=1))

def student_loss(student_output, target):
    return torch.nn.functional.cross_entropy(student_output, target)

def total_loss(student_output, teacher_output, target, temperature, alpha):
    loss1 = distillation_loss(student_output, teacher_output, temperature)
    loss2 = student_loss(student_output, target)
    return alpha * loss1 + (1 - alpha) * loss2

temperature = 2.1
alpha = 0.5

def train_teacher():
    for i, (data, target) in enumerate(train_loader):
        target = torch.nn.functional.one_hot(target, num_classes=10).float()
        optimizer_teacher.zero_grad()
        output = teacher(data)
        loss_teacher = torch.mean(-torch.sum(target * torch.log(torch.softmax(output, dim=1)), dim=1))
        loss_teacher.backward()
        optimizer_teacher.step()
        if i % 50 == 0:
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == torch.argmax(target, dim=1)).sum().item()
            accuracy = 100 * correct / target.size(0)
            print(f"Step {i}, Training Accuracy {accuracy}")
    # torch.save(teacher.state_dict(), './models/teacher1.pth')

def train_student():
    for i, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        optimizer_student.zero_grad()
        student_output = student(data)
        with torch.no_grad():
            teacher_output = teacher(data.view(-1, 1, 28, 28))
        loss_student = total_loss(student_output, teacher_output, target, temperature, alpha)
        loss_student.backward()
        optimizer_student.step()
        if i % 50 == 0:
            _, predicted = torch.max(student_output.data, 1)
            correct = (predicted == target).sum().item()
            accuracy = 100 * correct / target.size(0)
            print(f"Step {i}, Training Accuracy {accuracy}")
    # torch.save(student.state_dict(), './models/student.pth')

def test_model(model, test_loader, is_teacher=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            if is_teacher:
                output = model(data)
            else:
                data = data.view(data.size(0), -1)
                output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy of the {'Teacher' if is_teacher else 'Student'} Model is {accuracy}%")
    return accuracy

print("Teacher Training Started...")
train_teacher()
print("Teacher Training Ended...")
print("Student Training Started...")
train_student()
print("Student Training Ended...")

teacher_accuracy = test_model(teacher, test_loader, is_teacher=True)
student_accuracy = test_model(student, test_loader, is_teacher=False)