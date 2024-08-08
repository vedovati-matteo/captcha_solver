from __future__ import annotations
import os
import random
import typing as t
from PIL.Image import new as createImage, Image, QUAD, BILINEAR
from PIL.ImageDraw import Draw, ImageDraw
from PIL.ImageFilter import SMOOTH
from PIL.ImageFont import FreeTypeFont, truetype
from io import BytesIO

ColorTuple = t.Union[t.Tuple[int, int, int], t.Tuple[int, int, int, int]]
DEFAULT_FONTS = ['DroidSansMono.ttf']

def random_color(
    start: int,
    end: int,
    opacity: int | None = None) -> ColorTuple:
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return red, green, blue
    return red, green, blue, opacity

class Captcha:
    
    lookup_table: list[int] = [int(i * 1.97) for i in range(256)]
    character_offset_dx: tuple[int, int] = (0, 4)
    character_offset_dy: tuple[int, int] = (0, 6)
    character_rotate: tuple[int, int] = (-30, 30)
    character_warp_dx: tuple[float, float] = (0.1, 0.3)
    character_warp_dy: tuple[float, float] = (0.2, 0.3)
    word_space_probability: float = 0.5
    word_offset_dx: float = 0.25
    
    def __init__(
            self,
            #width: int = 160,
            height: int = 60,
            fonts: list[str] | None = None,
            font_sizes: tuple[int, ...] | None = None):
        #self.width = width
        self.height = height
        self.fonts = fonts or DEFAULT_FONTS
        self.font_sizes = font_sizes or (42, 50, 56)
        self.truefonts: list[FreeTypeFont] = [ truetype(n, s) for n in self.fonts for s in self.font_sizes ]
    
    @staticmethod
    def create_noise_curve(image: Image, color: ColorTuple) -> Image:
        w, h = image.size
        x1 = random.randint(0, int(w / 5))
        x2 = random.randint(w - int(w / 5), w)
        y1 = random.randint(int(h / 5), h - int(h / 5))
        y2 = random.randint(y1, h - int(h / 5))
        points = [x1, y1, x2, y2]
        end = random.randint(160, 200)
        start = random.randint(0, 20)
        Draw(image).arc(points, start, end, fill=color)
        return image

    @staticmethod
    def create_noise_dots(
            image: Image,
            color: ColorTuple,
            width: int = 3,
            number: int = 30) -> Image:
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image
    
    @staticmethod
    def show_image(image: Image, bounding_boxes: list[dict[str, t.Any]] = None) -> Image:
        if bounding_boxes is None:
            image.show()
        else:
            draw = Draw(image)
            for bb in bounding_boxes:
                draw.rectangle(bb["bb"], outline=(255, 0, 0), width=2)
            image.show()
    
    def _draw_character(
            self,
            c: str,
            color: ColorTuple) -> Image:
        font = random.choice(self.truefonts)
        
        temp_image = createImage('RGBA', (1, 1))
        temp_draw = Draw(temp_image)
        
        _, _, w, h = temp_draw.multiline_textbbox((1, 1), c, font=font)

        dx1 = random.randint(*self.character_offset_dx)
        dy1 = random.randint(*self.character_offset_dy)
        im = createImage('RGBA', (w + dx1, h + dy1))
        Draw(im).text((dx1, dy1), c, font=font, fill=color)

        # rotate
        im = im.crop(im.getbbox())
        im = im.rotate(
            random.uniform(*self.character_rotate),
            BILINEAR,
            expand=True,
        )

        
        # warp
        dx2 = w * random.uniform(*self.character_warp_dx)
        dy2 = h * random.uniform(*self.character_warp_dy)
        x1 = int(random.uniform(-dx2, dx2))
        y1 = int(random.uniform(-dy2, dy2))
        x2 = int(random.uniform(-dx2, dx2))
        y2 = int(random.uniform(-dy2, dy2))
        w2 = w + abs(x1) + abs(x2)
        h2 = h + abs(y1) + abs(y2)
        data = (
            x1, y1,
            -x1, h2 - y2,
            w2 + x2, h2 + y2,
            w2 - x2, -y1,
        )
        im = im.resize((w2, h2))
        im = im.transform((w, h), QUAD, data)
        
        return im

    def create_captcha_image(
            self,
            chars: str,
            color: ColorTuple,
            background: ColorTuple) -> Image:
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        
        images: list[Image] = []
        for c in chars:
            if random.random() > self.word_space_probability:
                images.append({"char": " ", "im": self._draw_character(" ", color)})
            images.append({"char": c, "im": self._draw_character(c, color)})

        width = sum([im_c["im"].size[0] for im_c in images]) # lenght for putting all char together
        
        image = createImage('RGB', (width, self.height), background)

        average = int(width / len(chars)) # average width of each char
        rand = int(self.word_offset_dx * average) # random offset
        offset = int(average * 0.1) # initial offset

        bounding_boxes = []
                
        for im_c in images:
            c = im_c["char"]
            im = im_c["im"]
            w, h = im.size
            
            mask = im.convert('L').point(self.lookup_table)
            
            # Save current offset
            current_offset = offset
            
            image.paste(im, (offset, int((self.height - h) / 2)), mask)
            
            # Save bounding box (left, upper, right, lower)
            if c != " ":  # Check if the image is not a space
                bounding_boxes.append({"char": c, "bb": (current_offset, int((self.height - h) / 2), current_offset + w, int((self.height + h) / 2))})
            
            offset = offset + w + random.randint(-rand, 0)

        return image, bounding_boxes
    
    def generate_image(self, chars: str) -> Image:
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = random_color(238, 255)
        color = random_color(10, 200, random.randint(220, 255))
        im, bb = self.create_captcha_image(chars, color, background)
        self.create_noise_dots(im, color)
        self.create_noise_curve(im, color)
        im = im.filter(SMOOTH)
        return im, bb
    
    def generate(self, chars: str, format: str = 'png') -> BytesIO:
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im, bb = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out, bb

    def write(self, chars: str, output: str, format: str = 'png') -> None:
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        im, bb = self.generate_image(chars)
        im.save(output, format=format)