import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as fun

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

'''
class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		self.hidden_layer = nn.Linear(178, 16)
		self.output_layer = nn.Linear(16, 5)

	def forward(self, x):
		x = torch.sigmoid(self.hidden_layer(x))
		x = self.output_layer(x)
		return x
'''

class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		self.hidden_layer = nn.Linear(178, 64)
		self.output_layer = nn.Linear(64, 5)
		self.dropout = nn.Dropout(0.4)
		self.batchNorm = nn.BatchNorm1d(178)

	def forward(self, x):
		x = fun.relu(self.dropout(self.hidden_layer(self.batchNorm(x))))
		x = self.output_layer(x)
		return x

'''
class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)
		self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
		self.fc2 = nn.Linear(in_features=128, out_features=5)

	def forward(self, x):
		x = self.pool(fun.relu(self.conv1(x)))
		x = self.pool(fun.relu(self.conv2(x)))
		x = x.view(-1, 16 * 41)
		x = fun.relu(self.fc1(x))
		x = self.fc2(x)
		return x
'''

class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)
		self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
		self.fc2 = nn.Linear(in_features=128, out_features=5)
		self.dropout = nn.Dropout(0.3)

	def forward(self, x):
		x = self.pool(fun.relu(self.dropout(self.conv1(x))))
		x = self.pool(fun.relu(self.dropout(self.conv2(x))))
		x = x.view(-1, 16 * 41)
		x = fun.relu(self.dropout(self.fc1(x)))
		x = self.fc2(x)
		return x

'''
class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.gru = nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
		self.fc = nn.Linear(in_features=16, out_features=5)

	def forward(self, x):
		x, _ = self.gru(x)
		x = self.fc(x[:, -1, :])
		return x
'''

class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.gru = nn.GRU(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
		self.fc1 = nn.Linear(in_features=32, out_features=16)
		self.fc2 = nn.Linear(in_features=16, out_features=5)

	def forward(self, x):
		x, _ = self.gru(x)
		x = self.fc1(x[:, -1, :])
		x = self.fc2(x)
		return x

'''
class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		self.fc1 = nn.Linear(in_features=dim_input, out_features=32)
		self.gru = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True)
		self.fc2 = nn.Linear(in_features=16, out_features=2)

	def forward(self, input_tuple):
		seqs, lengths = input_tuple
		seqs = torch.tanh(self.fc1(seqs))
		seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
		seqs, _ = self.gru(seqs)
		seqs, _ = pad_packed_sequence(seqs, batch_first=True)
		seqs = self.fc2(seqs[:, -1, :])
		return seqs
'''

class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		self.fc1 = nn.Linear(in_features=dim_input, out_features=128)
		self.gru = nn.GRU(input_size=128, hidden_size=32, num_layers=1, batch_first=True)
		self.fc2 = nn.Linear(in_features=32, out_features=8)
		self.fc3 = nn.Linear(in_features=8, out_features=2)

	def forward(self, input_tuple):
		seqs, lengths = input_tuple
		seqs = torch.tanh(self.fc1(seqs))
		seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
		seqs, _ = self.gru(seqs)
		seqs, _ = pad_packed_sequence(seqs, batch_first=True)
		seqs = self.fc2(torch.tanh(seqs[:, -1, :]))
		seqs = self.fc3(seqs)
		return seqs