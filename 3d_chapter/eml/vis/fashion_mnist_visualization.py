from config import cfg

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from datetime import datetime


class Fashion_mnist_visualization:

    def __init__(self):
        self.classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                        "Sneaker", "Bag", "Ankle boot", ]
        self.PDF_ROOT = cfg["paths"]["pdf"]

    # Converts an Fashion MNIST numeric id to a string.
    def to_label(self, id):
        return self.classes[id]

    # Applies the model to the data and plots the data.
    #  @param offset of the first image.
    #  @param stride stride between the images.
    #  @param dataloader from which the data is retrieved.
    #  @param model model which is used for the predictions.
    def plot(self, offset, stride, test_data, model, info):

        name = "Eval_" + datetime.today().strftime('%Y-%m-%d_%H:%M') + ".pdf"

        model.eval()

        with PdfPages(self.PDF_ROOT+name) as pdf:

            print(f"Creating pdf {self.PDF_ROOT+name} ...")

            for img_idx in range(offset, len(test_data.data)-1, stride):

                x, Y = test_data[img_idx][0], test_data[img_idx][1]

                prediction = model(x).argmax(1)

                plt.title("Is " + self.to_label(Y) +
                          " and predicted " + self.to_label(prediction))

                if img_idx == offset:
                    plt.xlabel(info)

                plt.imshow(x[0])
                pdf.savefig()
                plt.close(x)

            print(f"Sucesfully saved {self.PDF_ROOT+name}")
