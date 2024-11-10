import qiskit
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

# Importing Pillow for basic image manipulation
from PIL import Image

# Enabling Jupyter cell displays
from IPython.display import display

import math
import numpy as np

def generate_stripe_image(width, height):
    my_image = Image.new('L', (width, height))

    # Filling the image
    for x in range(width):
        for y in range(height):
            if x % 2 != 0:
                intensity = 50
            else:
                intensity = 255
            my_image.putpixel((x, y), intensity)
    return my_image


def get_intensities(image_name):
    # Opening the image and converting it to grayscale
    image = Image.open(image_name).convert('L')
    pixels = image.load()
    pixel_intensities = []
    
    # Get image dimensions
    width, height = image.size
    for x in range(width):
        for y in range(height):
            intensity = image.getpixel((x, y))
            pixel_intensities.append(intensity / 255)
    # display(image)
    return pixel_intensities

def normalize(intensities):
    squared_intensities = []
    normalized_intensities = []
    for i in intensities:
        squared_intensities.append(i**2)
    square_total = sum(squared_intensities)
    for i in intensities:
        normalized_intensities.append(i / math.sqrt(square_total))
    return normalized_intensities, square_total

# Defining a function that runs a simulation of a given circuit
def simulate(circuit, shot_number):
    backend = AerSimulator()
    transpiled_circuit = transpile(circuit, backend)
    result = backend.run(transpiled_circuit, shots=shot_number).result()
    counts = result.get_counts()
    return counts

# initial_circuit() initializes a circuit with an ancillary qubit and an
# arbitrary number of data qubits set to an arbitrary state
def initial_circuit(coeffs):
    
    # Bit count is all data qubits plus an ancillary qubit
    bit_count = int(math.log2(len(coeffs)) + 1)
    
    # State vector is all data qubits without the ancillary qubit
    state_vec = np.array(coeffs, dtype=complex)
    
    # Building data qubit indices
    qubit_indices  = []
    for n in range(1, bit_count):
        qubit_indices.append(n)
    print(qubit_indices)
    
    # Initializing the quantum circuit
    qc = QuantumCircuit(bit_count, bit_count)
    qc.initialize(state_vec, qubit_indices)
    return qc


def extract_intensity(counts, shots, square_total):
    counts_list = list(counts.values())
    print(counts_list)
    probabilities = []
    coeffs = []
    intensities = []
    for c in counts_list:
        probabilities.append(c / shots)
    for p in probabilities:
        coeffs.append(math.sqrt(p))
    for c in coeffs:
        intensity = c * math.sqrt(square_total) * 255
        if intensity <= 255:
            intensities.append(int(intensity))
        else:
            intensities.append(255)
    
    return intensities


def rebuild_image(width, height, extracted_intensities):
    image_dim = width
    
    # Create a new image with grayscale mode 'L'
    image_canvas = Image.new('L', (image_dim, image_dim))
    x_coord = 0
    y_coord = 0
    
    
    extracted_index = 0
    for x in range(width):
        for y in range(height):
            image_canvas.putpixel((x, y), extracted_intensities[extracted_index])
            extracted_index += 1
            
    return image_canvas
