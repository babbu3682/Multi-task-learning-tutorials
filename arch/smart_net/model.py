from typing import Optional, Union, List

from .seg_decoder import Unet_Decoder
from .rec_decoder import AE_Decoder


from ..encoders import get_encoder
from ..base import SegmentationHead, ClassificationHead, ReconstructionHead
from ..base import Single_Task_Cls_Model, Single_Task_Seg_Model, Multi_Task_1_Model, Multi_Task_2_Model


## STL #1
class STL_1_Net(Single_Task_Cls_Model):
    def __init__(
        self,
        encoder_name: str = "resnet50",  
        encoder_depth: int = 5,
        in_channels: int = 1,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
        )

        # CLS
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1], 
            pooling='avg', 
            dropout=0.5,
            out_channels=1,
        )

        self.name = "MTL-Net-{}".format(encoder_name)
        self.initialize()

## STL #2
class STL_2_Net(Single_Task_Seg_Model):
    def __init__(
        self,
        encoder_name: str = "resnet50",  
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 1,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
        )


        # SEG
        self.seg_decoder = Unet_Decoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1]//2,
            out_channels=1,
            kernel_size=3,
        )


        self.name = "MTL-Net-{}".format(encoder_name)
        self.initialize()


## MTL #1
class MTL_1_Net(Multi_Task_1_Model):
    def __init__(
        self,
        encoder_name: str = "resnet50",  
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 1,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
        )

        # CLS
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1], 
            pooling='avg', 
            dropout=0.5,
            out_channels=1,
        )

        # SEG
        self.seg_decoder = Unet_Decoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1]//2,
            out_channels=1,
            kernel_size=3,
        )


        self.name = "MTL-Net-{}".format(encoder_name)
        self.initialize()


## MTL #2
class MTL_2_Net(Multi_Task_2_Model):
    def __init__(
        self,
        encoder_name: str = "resnet50",  
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 1,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
        )

        # CLS
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1], 
            pooling='avg', 
            dropout=0.5,
            out_channels=1,
        )

        # SEG
        self.seg_decoder = Unet_Decoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1]//2,
            out_channels=1,
            kernel_size=3,
        )

        # REC
        self.rec_decoder = AE_Decoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            use_batchnorm=False,
            center=False,
            attention_type=decoder_attention_type,
        )
        self.reconstruction_head = ReconstructionHead(
            in_channels=decoder_channels[-1]//2,
            out_channels=1,
            kernel_size=3,
        )


        self.name = "MTL-Net-{}".format(encoder_name)
        self.initialize()
