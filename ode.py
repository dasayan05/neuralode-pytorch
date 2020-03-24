import torch, torchvision
import torch.nn.functional as F

from neuralode import ODELayer

class Dynamics(torch.nn.Module):
    # Defines the dynamics of ODE
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim
        self.linear = torch.nn.Linear(self.n_dim, self.n_dim)
    def forward(self, z, t):
        return torch.tanh(self.linear(z)) * t

class ODEClassifier(torch.nn.Module):
    # The whole model containing a ODELayer and a classifier
    def __init__(self, n_dim, n_classes, ode_dynamics, t_start = 0., t_end = 1., granularity = 50):
        super().__init__()
        self.n_dim, self.n_classes = n_dim, n_classes
        self.t_start, self.t_end, self.granularity = t_start, t_end, granularity

        self.odelayer = ODELayer(ode_dynamics, self.t_start, self.t_end, self.granularity)
        self.classifier = torch.nn.Linear(self.n_dim, self.n_classes)

    def forward(self, x, drop_ode=False):
        if not drop_ode:
            x = self.odelayer(x)
        return F.log_softmax(self.classifier(x), 1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, required=False, default=1e-3, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=100, help='Total epochs')
    parser.add_argument('--drop_ode', action='store_true', help='Just drop the ODE Layer (for comparison)')
    args = parser.parse_args()

    dynamics = Dynamics(28 * 28)
    model = ODEClassifier(28 * 28, 10, dynamics)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    mnist = torchvision.datasets.MNIST('./mnist', download=True, train=True, transform=torchvision.transforms.ToTensor())
    mnisttest = torchvision.datasets.MNIST('./mnist', download=True, train=False, transform=torchvision.transforms.ToTensor())
    mnistdl = torch.utils.data.DataLoader(mnist, shuffle=True, batch_size=args.batch_size, drop_last=True, pin_memory=True)
    mnisttestdl = torch.utils.data.DataLoader(mnisttest, shuffle=True, batch_size=args.batch_size, drop_last=True, pin_memory=True)

    for e in range(args.epochs):
        for i, (X, Y) in enumerate(mnistdl):
            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()
            X = X.view(args.batch_size, -1) # flatten everything
            
            output = model(X, drop_ode=args.drop_ode)
            loss = F.nll_loss(output, Y)
            
            if i % 20 == 0:
                print(f'[Training] {i}/{e}/{args.epochs} -> Loss: {loss.item()}')

            optim.zero_grad()
            loss.backward()
            optim.step()

        total, correct = 0, 0
        for j, (X, Y) in enumerate(mnisttestdl):
            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()
            X = X.view(args.batch_size, -1) # flatten everything
            
            output = model(X, drop_ode=args.drop_ode)
            correct += (output.argmax(1) == Y).sum().item()
            total += args.batch_size

        accuracy = (correct / float(total)) * 100
        print(f'[Testing] -/{e}/{args.epochs} -> Accuracy: {accuracy}')