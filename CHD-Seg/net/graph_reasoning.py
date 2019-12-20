import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    def __init__(self, num_channel):
        super(GraphConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_channel, num_channel, kernel_size=1),
            nn.BatchNorm1d(num_channel),                       #### BN
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x.contiguous())

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h

class GraphReasoning(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(GraphReasoning, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.state_channels = int(self.out_channels / 2)
        self.node_num = int(self.out_channels / 4)
        
        self.conv_state = nn.Conv2d(self.in_channels, self.state_channels, kernel_size=1)
        self.conv_proj = nn.Conv2d(self.in_channels, self.node_num, kernel_size=1)
        self.conv_rproj = nn.Conv2d(self.in_channels, self.node_num, kernel_size=1)
        # self.graph_conv = GraphConvolution(self.state_channels)
        self.graph_conv = GCN(self.state_channels, self.node_num)
        self.conv_extend = nn.Conv2d(self.state_channels, self.out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(self.out_channels, eps=1e-04)

    def forward(self, x):
        n = x.size(0)
        x_state_reshaped = self.conv_state(x).view(n, self.state_channels, -1)
        x_proj_reshaped = self.conv_proj(x).view(n, self.node_num, -1)
        # x_rproj_reshaped = x_proj_reshaped
        x_rproj_reshaped = self.conv_rproj(x).view(n, self.node_num, -1)

        x_node = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_gragh_conv = self.graph_conv(x_node)
        x_out = torch.matmul(x_gragh_conv, x_rproj_reshaped)
        x_out = x_out.view(n, self.state_channels, *x.size()[2:])
        x_out = x + self.bn(self.conv_extend(x_out))

        return x_out

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class GraphReasoningASP(nn.Module):
    def __init__(self, in_channels, out_channels=None, dilations=(2, 4, 6)):
        super(GraphReasoningASP, self).__init__()
        if out_channels == None:
            out_channels = in_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        modules.append(ASPPConv(in_channels, out_channels, dilations[0]))
        modules.append(ASPPConv(in_channels, out_channels, dilations[1]))
        modules.append(ASPPConv(in_channels, out_channels, dilations[2]))
        modules.append(GraphReasoning(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

if __name__ == "__main__":
    data = torch.autograd.Variable(torch.randn(2, 512, 32, 32))
    net = GraphReasoningASP(512)
    print(net)
    print(net(data).size())




