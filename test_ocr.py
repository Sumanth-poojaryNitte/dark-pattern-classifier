import easyocr

print("Loading EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("EasyOCR loaded successfully!")

image_path = "dark_pattern_scarcity_example.png"

print("\nReading image...")

results = reader.readtext(image_path)

print("\n========== OCR RESULT ==========")

if not results:
    print("No text detected.")
else:
    for detection in results:
        text = detection[1]
        confidence = detection[2]

        print(f"Text: {text}")
        print(f"OCR Confidence: {confidence:.2f}")
        print("-" * 50)