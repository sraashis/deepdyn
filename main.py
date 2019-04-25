import testarch.unet as unet
import testarch.unet.runs as r_unet
import testarch.miniunet as mini_unet
import testarch.miniunet.runs as r_miniunet
import torchvision.transforms as tmf


transforms = tmf.Compose([
    tmf.ToPILImage(),
    tmf.ToTensor()
])

if __name__ == "__main__":
    unet.run([r_unet.DRIVE], transforms)
    mini_unet.run([r_miniunet.DRIVE], transforms)

