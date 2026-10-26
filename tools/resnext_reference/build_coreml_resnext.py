"""
Buduje ASLClassifier.mlpackage dla aplikacji: ResNeXt50 (ImageNet) → embedding 2048 → regresja logistyczna → softmax.
Odwzorowuje notebook „f1 > 0.99 with logistic regression and ResNeXt” (mikedelong) na Kaggle.

Dwa tryby:

  # 1) Masz już wytrenowaną regresję logistyczną (pickle/joblib sklearn LogisticRegression):
  python build_coreml.py --logreg logreg.pkl

  # 2) Wytrenuj od zera na datasecie „American Sign Language (ASL) Hand Skeleton Dataset”:
  python build_coreml.py --train /ścieżka/do/dataset --count 500

W trybie 1 zapisz model w notebooku na Kaggle:  import joblib; joblib.dump(logreg, 'logreg.pkl')

Wynik: ../ASSN/ASLClassifier.mlpackage (Xcode sam go dołączy i skompiluje).

Zależności: pip install torch torchvision coremltools scikit-learn pillow joblib numpy
"""

import argparse
from glob import glob
from os.path import basename, dirname, join, abspath

import joblib
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEFAULT_OUTPUT = join(dirname(abspath(__file__)), '..', '..', 'ASSN', 'ASLClassifier.mlpackage')


def backbone() -> torch.nn.Module:
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()  # wyjście = avgpool spłaszczony do 2048, jak hook w notebooku
    return model.eval()


def train(dataset: str, count: int):
    """Dokładnie jak notebook: max `count` obrazków na literę, Resize(224) + ImageNet normalize, LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    model = backbone()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    features, labels = [], []
    for folder in sorted(glob(join(dataset, '*'))):
        files = sorted(glob(join(folder, '*.*')))[:count]
        batch = torch.stack([preprocess(Image.open(f).convert('RGB')) for f in files])
        with torch.no_grad():
            for chunk in torch.split(batch, 64):
                features.append(model(chunk).numpy())
        labels += [basename(folder)] * len(files)
        print(f'{basename(folder)}: {len(files)} obrazków')
    x = np.concatenate(features)

    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, test_size=0.25, random_state=2025, stratify=labels)
    logreg = LogisticRegression(max_iter=1000, tol=1e-12).fit(x_train, y_train)
    print('f1 (test): {:.4f}'.format(f1_score(y_test, logreg.predict(x_test), average='weighted')))
    joblib.dump(logreg, 'logreg.pkl')
    return logreg


class ASLClassifier(torch.nn.Module):
    """Wejście: obraz RGB 0…1 (Core ML skaluje 0…255 × 1/255). Normalizacja ImageNet jest w środku modelu."""

    def __init__(self, logreg):
        super().__init__()
        self.backbone = backbone()
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1))
        coef = np.asarray(logreg.coef_, dtype=np.float32)
        if coef.shape[0] == 1:
            raise SystemExit('Model binarny – oczekiwano wieloklasowej regresji logistycznej.')
        self.head = torch.nn.Linear(coef.shape[1], coef.shape[0])
        self.head.weight.data = torch.from_numpy(coef)
        self.head.bias.data = torch.from_numpy(np.asarray(logreg.intercept_, dtype=np.float32))

    def forward(self, image):
        x = (image - self.mean) / self.std
        # sklearn (lbfgs, multinomial) liczy predict_proba jako softmax(coef·x + b).
        return torch.softmax(self.head(self.backbone(x)), dim=1)


def export(logreg, output: str):
    import coremltools as ct

    model = ASLClassifier(logreg).eval()
    traced = torch.jit.trace(model, torch.rand(1, 3, 224, 224))
    labels = [str(c) for c in logreg.classes_]
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name='image', shape=(1, 3, 224, 224), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        classifier_config=ct.ClassifierConfig(labels),
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.short_description = 'ASL A–Z ze szkieletu dłoni (ResNeXt50 + regresja logistyczna)'
    mlmodel.save(output)
    print(f'Zapisano {abspath(output)} ({len(labels)} klas: {" ".join(labels)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--logreg', help='plik .pkl/.joblib z sklearn LogisticRegression')
    source.add_argument('--train', help='folder dataset/ z podfolderami A…Z')
    parser.add_argument('--count', type=int, default=500, help='obrazków na literę (notebook: 500)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    logreg = joblib.load(args.logreg) if args.logreg else train(args.train, args.count)
    export(logreg, args.output)
