import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

cm = np.array([
    [13298, 41],
    [1675, 35]
])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not readmitted", "Readmitted"]
)

display.plot(values_format="d")
plt.title("Logistic Regression — Validation Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_logistic_regression.png", dpi=300)
plt.show()