#!/usr/bin/env python3
"""
Create app icon for the Translator app.
Generates a modern, clean icon with a translation/speech bubble design.
"""

import subprocess
import tempfile
import os
import math
from pathlib import Path

def create_icon_with_pillow(output_png: str, size: int = 1024):
    """Create icon using Pillow."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create image with gradient-like background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Background colors (purple gradient simulation)
    bg_color1 = (99, 102, 241)   # Indigo
    bg_color2 = (139, 92, 246)   # Purple
    
    # Draw rounded rectangle background
    padding = int(size * 0.04)
    corner_radius = int(size * 0.2)
    
    # Create gradient effect by drawing multiple rectangles
    for i in range(size - 2 * padding):
        progress = i / (size - 2 * padding)
        r = int(bg_color1[0] + (bg_color2[0] - bg_color1[0]) * progress)
        g = int(bg_color1[1] + (bg_color2[1] - bg_color1[1]) * progress)
        b = int(bg_color1[2] + (bg_color2[2] - bg_color1[2]) * progress)
        
        y = padding + i
        draw.line([(padding, y), (size - padding, y)], fill=(r, g, b, 255))
    
    # Draw rounded corners mask
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        [padding, padding, size - padding, size - padding],
        radius=corner_radius,
        fill=255
    )
    
    # Apply mask
    img.putalpha(mask)
    
    # Draw speech bubble
    bubble_margin = int(size * 0.15)
    bubble_bottom = int(size * 0.6)
    bubble_radius = int(size * 0.06)
    
    # White bubble with shadow effect
    shadow_offset = int(size * 0.015)
    
    # Shadow
    shadow = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        [bubble_margin + shadow_offset, bubble_margin + shadow_offset, 
         size - bubble_margin + shadow_offset, bubble_bottom + shadow_offset],
        radius=bubble_radius,
        fill=(0, 0, 0, 60)
    )
    
    # Bubble tail shadow
    tail_points = [
        (int(size * 0.25), bubble_bottom + shadow_offset),
        (int(size * 0.35), bubble_bottom + shadow_offset),
        (int(size * 0.2), int(size * 0.72) + shadow_offset)
    ]
    shadow_draw.polygon(tail_points, fill=(0, 0, 0, 60))
    
    img = Image.alpha_composite(img, shadow)
    draw = ImageDraw.Draw(img)
    
    # Main bubble
    draw.rounded_rectangle(
        [bubble_margin, bubble_margin, size - bubble_margin, bubble_bottom],
        radius=bubble_radius,
        fill=(255, 255, 255, 255)
    )
    
    # Bubble tail
    tail_points = [
        (int(size * 0.25), bubble_bottom - 2),
        (int(size * 0.35), bubble_bottom - 2),
        (int(size * 0.2), int(size * 0.70))
    ]
    draw.polygon(tail_points, fill=(255, 255, 255, 255))
    
    # Audio waveform bars
    bar_color = (99, 102, 241)  # Purple
    bar_width = int(size * 0.035)
    bar_gap = int(size * 0.065)
    bar_start_x = int(size * 0.22)
    bar_center_y = int(size * 0.38)
    
    # Wave heights (normalized)
    wave_heights = [0.4, 0.7, 0.5, 0.9, 0.6, 0.3]
    max_bar_height = int(size * 0.18)
    
    for i, height in enumerate(wave_heights):
        x = bar_start_x + i * bar_gap
        bar_height = int(max_bar_height * height)
        y1 = bar_center_y - bar_height // 2
        y2 = bar_center_y + bar_height // 2
        
        # Draw rounded bar
        draw.rounded_rectangle(
            [x, y1, x + bar_width, y2],
            radius=bar_width // 2,
            fill=bar_color
        )
    
    # Globe icon (translation symbol)
    globe_center = (int(size * 0.7), int(size * 0.78))
    globe_radius = int(size * 0.1)
    globe_color = (255, 255, 255)
    line_width = int(size * 0.012)
    
    # Globe circle
    draw.ellipse(
        [globe_center[0] - globe_radius, globe_center[1] - globe_radius,
         globe_center[0] + globe_radius, globe_center[1] + globe_radius],
        outline=globe_color, width=line_width
    )
    
    # Horizontal line
    draw.line(
        [globe_center[0] - globe_radius, globe_center[1],
         globe_center[0] + globe_radius, globe_center[1]],
        fill=globe_color, width=line_width
    )
    
    # Vertical line
    draw.line(
        [globe_center[0], globe_center[1] - globe_radius,
         globe_center[0], globe_center[1] + globe_radius],
        fill=globe_color, width=line_width
    )
    
    # Curved lines for globe effect
    draw.arc(
        [globe_center[0] - globe_radius * 0.5, globe_center[1] - globe_radius,
         globe_center[0] + globe_radius * 0.5, globe_center[1] + globe_radius],
        0, 360, fill=globe_color, width=line_width // 2
    )
    
    # Headphones icon
    hp_center = (int(size * 0.3), int(size * 0.78))
    hp_radius = int(size * 0.08)
    hp_color = (255, 255, 255)
    hp_width = int(size * 0.015)
    
    # Headband arc
    draw.arc(
        [hp_center[0] - hp_radius, hp_center[1] - hp_radius,
         hp_center[0] + hp_radius, hp_center[1] + int(hp_radius * 0.3)],
        180, 360, fill=hp_color, width=hp_width
    )
    
    # Ear cups
    cup_size = int(size * 0.035)
    # Left cup
    draw.rounded_rectangle(
        [hp_center[0] - hp_radius - cup_size // 2, hp_center[1] - cup_size // 2,
         hp_center[0] - hp_radius + cup_size // 2, hp_center[1] + cup_size],
        radius=cup_size // 4,
        fill=hp_color
    )
    # Right cup
    draw.rounded_rectangle(
        [hp_center[0] + hp_radius - cup_size // 2, hp_center[1] - cup_size // 2,
         hp_center[0] + hp_radius + cup_size // 2, hp_center[1] + cup_size],
        radius=cup_size // 4,
        fill=hp_color
    )
    
    img.save(output_png, 'PNG')
    return True


def create_icns(output_path: str):
    """Create .icns file."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        iconset_path = os.path.join(tmpdir, "icon.iconset")
        os.makedirs(iconset_path)
        
        # Create base PNG using Pillow
        base_png = os.path.join(tmpdir, "base_1024.png")
        
        try:
            create_icon_with_pillow(base_png, 1024)
            print("✅ Created icon using Pillow")
        except ImportError:
            print("⚠️  Pillow not installed. Install with: pip install pillow")
            print("    Creating minimal fallback icon...")
            create_minimal_fallback(base_png)
        
        # Generate all required sizes using sips
        sizes = [16, 32, 64, 128, 256, 512]
        
        for size in sizes:
            # Standard resolution
            out_file = os.path.join(iconset_path, f"icon_{size}x{size}.png")
            subprocess.run([
                'sips', '-z', str(size), str(size), base_png,
                '--out', out_file
            ], capture_output=True)
            
            # @2x retina
            out_file_2x = os.path.join(iconset_path, f"icon_{size}x{size}@2x.png")
            subprocess.run([
                'sips', '-z', str(size * 2), str(size * 2), base_png,
                '--out', out_file_2x
            ], capture_output=True)
        
        # Convert iconset to icns
        result = subprocess.run([
            'iconutil', '-c', 'icns', iconset_path, '-o', output_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Created icon: {output_path}")
        else:
            print(f"❌ Failed to create icns: {result.stderr}")


def create_minimal_fallback(output_path: str):
    """Create a minimal PNG icon as fallback."""
    # Create a simple 1024x1024 purple square with 'T'
    # This is a last resort if Pillow isn't available
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGBA', (1024, 1024), (99, 102, 241, 255))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([40, 40, 984, 984], radius=200, fill=(99, 102, 241, 255))
        img.save(output_path, 'PNG')
    except ImportError:
        # Absolute minimal - just touch the file
        Path(output_path).touch()


if __name__ == "__main__":
    script_dir = Path(__file__).parent.parent
    output = script_dir / "assets" / "icon.icns"
    output.parent.mkdir(parents=True, exist_ok=True)
    create_icns(str(output))
