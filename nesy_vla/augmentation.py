"""Photometric/codec augmentation with unchanged navigation geometry."""
import io,random
from PIL import Image,ImageEnhance,ImageFilter


def perturb_observation(image,rng=random):
    image=image.convert('RGB');size=image.size
    # Different sensor transports must not change the learned predicate meaning.
    intermediate=rng.choice([size[0],228,232])
    result=image.resize((intermediate,intermediate),Image.Resampling.BILINEAR).resize(size,Image.Resampling.BILINEAR)
    if rng.random()<.75:
        buffer=io.BytesIO();result.save(buffer,format='JPEG',quality=rng.randint(82,99));buffer.seek(0)
        result=Image.open(buffer).convert('RGB')
    result=ImageEnhance.Brightness(result).enhance(rng.uniform(.96,1.04))
    result=ImageEnhance.Contrast(result).enhance(rng.uniform(.96,1.04))
    return result.filter(ImageFilter.GaussianBlur(rng.uniform(0,.25)))
