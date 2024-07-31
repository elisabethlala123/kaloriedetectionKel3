import torch
import torchvision  # Add this line to import the torchvision module
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')  

def load_model(model_path):
    # Memuat model yang sudah dilatih untuk klasifikasi dan mengembalikan hanya fitur-fiturnya
    backbone = resnet50(pretrained=True).eval()  # Menggunakan ResNet-50 sebagai backbone yang sudah dilatih
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # Menghapus layer FC dan layer avgpool terakhir
    backbone.out_channels = 2048  # Menetapkan jumlah output channel dari backbone

    # Menghasilkan anchor
    rpn_anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),  # Ukuran anchor yang dihasilkan
        aspect_ratios=((0.5, 1.0, 2.0),) * 5  # Rasio aspek dari anchor yang dihasilkan
    )

    # RoI Pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    # Membuat model Faster R-CNN
    model = FasterRCNN(backbone,
                       num_classes=8,  # Jumlah kelas + background (7 kelas + 1 background)
                       rpn_anchor_generator=rpn_anchor_generator,
                       box_roi_pool=roi_pooler)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

model = load_model('model_fastrcnn_resnet.pth')

