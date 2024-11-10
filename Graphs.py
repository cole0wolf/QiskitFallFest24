import matplotlib.pyplot as plt

dim = [24,32,64,128]
pixels = []
for i in range(4):
    pixels.append(dim[i] ** 2)
standard_image = [0.5625, 0.625, 0.646, 0.65]
classical_ed = [0.55, 0.6, 0.7, 0.85]
quantum_ed = [0.7, 0.9]


plt.plot(pixels, standard_image, label='Raw Image')
plt.plot(pixels, classical_ed, label='Classical Edge Detection')
plt.plot((pixels[0], pixels[1]), quantum_ed, label='Quantum Edge Detection')
plt.title('Speed Limit Sign Classification')
plt.xlabel('Image Size (pixels)')
plt.ylabel('Final Test Accuracy')
plt.legend(loc='best')
plt.savefig('ImageClassificationAccuracy.png')

plt.show()