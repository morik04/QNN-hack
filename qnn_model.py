# qnn_model.py
import torch
import torch.nn as nn
import pennylane as qml
import os

class QuantumCircuit:
    """Quantum circuit definition for the hybrid model"""
    
    def __init__(self, num_qubits=4, shots=100):
        self.num_qubits = num_qubits
        self.shots = shots
        self.device = self._setup_device()
    
    def _setup_device(self):
        try:
            return qml.device(
                "braket.aws.qubit",
                device_arn=os.getenv('BRAKET_DEVICE', 'arn:aws:braket:::device/quantum-simulator/amazon/sv1'),
                wires=self.num_qubits,
                shots=self.shots
            )
        except:
            return qml.device("default.qubit", wires=self.num_qubits, shots=self.shots)
    
    @property
    def circuit(self):
        @qml.qnode(self.device, interface="torch")
        def quantum_sub_circuit(inputs, weights):
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)
            for i in range(self.num_qubits):
                qml.RZ(weights[i], wires=i)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(self.num_qubits):
                qml.RY(weights[i + self.num_qubits], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        return quantum_sub_circuit


class HybridDensityQNN(nn.Module):
    """Hybrid CNN + Quantum Neural Network"""
    
    def __init__(self, num_sub_unitaries=2, num_qubits=4):
        super(HybridDensityQNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, num_qubits)
        
        self.K = num_sub_unitaries
        self.num_qubits = num_qubits
        
        quantum_circuit = QuantumCircuit(num_qubits=num_qubits).circuit
        self.quantum_layers = nn.ModuleList([
            qml.qnn.TorchLayer(quantum_circuit, {"weights": (num_qubits * 2,)})
            for _ in range(self.K)
        ])
        
        self.alpha = nn.Parameter(torch.ones(self.K))
        self.fc2 = nn.Linear(num_qubits, 10)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        
        quantum_outputs = []
        for i in range(batch_size):
            sample = x[i]
            circuit_outputs = [self.quantum_layers[k](sample) for k in range(self.K)]
            alpha_norm = torch.softmax(self.alpha, dim=0)
            weighted_out = sum(alpha_norm[k] * circuit_outputs[k] for k in range(self.K))
            quantum_outputs.append(weighted_out)
        
        quantum_out = torch.stack(quantum_outputs)
        out = self.fc2(quantum_out)
        return out
