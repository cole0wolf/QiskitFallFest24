# Importing standard Qiskit libraries and configuring account
from qiskit import *
from qiskit.compiler import transpile, assemble
# from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit_aer import Aer, AerSimulator


import matplotlib.pyplot as plt
import numpy as np

import os

# Importing Pillow for basic image manipulation
from PIL import Image

# Enabling Jupyter cell displays
from IPython.display import display

from QuantumImageMethods import *

source_folder = 'train/50'
target_folder = 'EdgeDetectedTrain50'

# Function for plotting the image using matplotlib
def plot_image(img, title: str):
    plt.title(title)
    plt.xticks(range(img.shape[0]))
    plt.yticks(range(img.shape[1]))
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
    plt.show()


def fileNames(dirName):
    directory = os.fsencode(dirName)
    names = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename != '.DS_Store':
            names.append(dirName + '/' + filename)
    return names
print(fileNames(source_folder))

#image_names = ['testsign.png']
image_names = fileNames(source_folder)
counter = 1
for image_name in image_names:

    width = 32
    height = 32

    my_image = generate_import_image(width, height, image_name)
    my_binary_image = make_binary(my_image, 128)
    display(my_image)
    array_image = np.array(my_binary_image)
    print(array_image)

    intensities = get_intensities(my_binary_image)

    # Normalizing    
    normalized_intensities, square_total = normalize(intensities)
    array_intensities = np.array(normalized_intensities)
    shaped_intensities = array_intensities.reshape(width, height)
    print("shaped intensities: " + str(shaped_intensities))

    qc = quantum_rep(normalized_intensities)
    print("normalized intensities: " + str(normalized_intensities))

    # Get the amplitude ancoded pixel values
    # Horizontal: Original image
    shaped_intensities_h = shaped_intensities

    # Vertical: Transpose of Original image
    shaped_intensities_v = shaped_intensities.T

    print("list shaped intensities: " + str(shaped_intensities_h))
    qc_intensities_h = []
    qc_intensities_v = []
    for element in shaped_intensities_h.flat:
        qc_intensities_h.append(element)
    for element in shaped_intensities_v.flat:
        qc_intensities_v.append(element)



    # Initialize some global variable for number of qubits
    data_qb = int(2*math.log2(width))
    anc_qb = 1
    total_qb = data_qb + anc_qb

    # Initialize the amplitude permutation unitary
    D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)

    # Create the circuit for horizontal scan
    qc_h = QuantumCircuit(total_qb)
    qc_h.initialize(qc_intensities_h, range(1, total_qb))
    qc_h.h(0)
    qc_h.unitary(D2n_1, range(total_qb))
    qc_h.h(0)
    display(qc_h.draw('mpl', fold=-1))

    # Create the circuit for vertical scan
    qc_v = QuantumCircuit(total_qb)
    qc_v.initialize(qc_intensities_v, range(1, total_qb))
    qc_v.h(0)
    qc_v.unitary(D2n_1, range(total_qb))
    qc_v.h(0)
    display(qc_v.draw('mpl', fold=-1))

    # Combine both circuits into a single list
    circ_list = [qc_h, qc_v]

    # Simulating the cirucits
    back = Aer.get_backend('statevector_simulator')
    results = back.run(circ_list, backend=back).result()
    sv_h = results.get_statevector(qc_h)
    sv_v = results.get_statevector(qc_v)

    from qiskit.visualization import array_to_latex
    print('Horizontal scan statevector:')
    #print(np.array(sv_h))
    display(array_to_latex(np.array(sv_h)[:30], max_size=30))
    print()
    print('Vertical scan statevector:')
    # display(array_to_latex(np.array(sv_v)[:30], max_size=30))

    # Classical postprocessing for plotting the output

    # Defining a lambda function for
    # thresholding to binary values
    threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

    # Selecting odd states from the raw statevector and
    # reshaping column vector of size 64 to an 8x8 matrix
    edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(width, height)
    edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(width, height).T

    # Plotting the Horizontal and vertical scans
    plot_image(edge_scan_h.T, 'Horizontal scan output')
    plot_image(edge_scan_v.T, 'Vertical scan output')

    # Combining the horizontal and vertical component of the result
    edge_scan_sim = edge_scan_h.T | edge_scan_v.T
    print("E: " + str(edge_scan_sim))

    # Plotting the original and edge-detected images
    plot_image(array_image, 'Original image')
    plot_image(edge_scan_sim, 'Edge Detected image')
    
    print("E(0): " + str(edge_scan_sim[0]))
    final_edge_image = Image.new('L', (width, height))
    for x in range(len(edge_scan_sim[0])):
        for y in range(len(edge_scan_sim[1])):
            if edge_scan_sim[x][y] > 0:
                final_edge_image.putpixel((x, y), 255)
            else:
                final_edge_image.putpixel((x, y), 0)


    os.makedirs(target_folder, exist_ok=True)
    output_path = os.path.join(target_folder, '50train' + str(counter) + '.jpg')
    final_edge_image.save(output_path)
    counter += 1
    display(final_edge_image)
