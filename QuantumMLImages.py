import qiskit
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.visualization import plot_histogram

# Importing Pillow for basic image manipulation
from PIL import Image

from QuantumImageMethods import *

# Enabling Jupyter cell displays
from IPython.display import display

import math
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------------------------
width = 32
height = 32

my_image = generate_stripe_image(width, height)

my_image.save('my_image.jpg')

image_name = 'my_image.jpg'
intensities = get_intensities(image_name)

# Normalizing    
normalized_intensities, square_total = normalize(intensities)

# Checking
print(normalized_intensities)
square_intensities = []
for n in normalized_intensities:
    square_intensities.append(n**2)
print(sum(square_intensities))
print(square_total)


# Encoding the 2x2 image into a quantum state
# c0|00> + c1|01> + c2|10> + c3|00>
# Initialize state with c0 = c1 = c2 = c3 = 1/2 since all pixel intensities are the same 
coeffs = []
for ni in normalized_intensities:
    coeffs.append(ni)

# Adding measurement to each data qubit
qc = initial_circuit(coeffs)
for i in range(1, qc.num_qubits):
    qc.measure(i, i)


# Checking the initialization
shot_number = 1000000
simulation_counts = simulate(qc, shot_number)
#print(simulation_counts)
sorted_sim_counts = dict(sorted(simulation_counts.items()))
#print(sorted_sim_counts)
display(plot_histogram(sorted_sim_counts))

# Counts to intensity
# Black: 0, White: 255
extracted_intensities = extract_intensity(sorted_sim_counts, shot_number, square_total)

# Temporary fix for falling short in intensity values
while len(extracted_intensities) < width * height:
    extracted_intensities.append(0)
    
# print(len(extracted_intensities))
# print(extracted_intensities)

extracted_image = rebuild_image(width, height, extracted_intensities)
display(extracted_image)
extracted_image.save('Extraction.jpg')

# Analysis
pixel_count = width * height
intensity_errors = []
total_accuracy = 0
total_error = 0
index = 0
for e in extracted_intensities:
    error = abs(intensities[index] * 255 - e) / (intensities[index] * 255)
    accuracy = 1 - error
    intensity_errors.append(error)
    total_accuracy += accuracy
    total_error += error
    index += 1

percent_accuracy = round((total_accuracy / pixel_count) * 100, 3)
percent_error = round((total_error / pixel_count) * 100, 3)
# print("Pixel errors: " + str(intensity_errors))
print("Total accuracy: " + str(round(percent_accuracy, 3)) + "%")
print("Total error: " + str(percent_error))
