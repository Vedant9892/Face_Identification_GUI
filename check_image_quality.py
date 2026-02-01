import os
import cv2
from mtcnn import MTCNN
from config import FACE_IMAGES_DIR


def check_image_quality():
    if not os.path.exists(FACE_IMAGES_DIR):
        print(f"Error: {FACE_IMAGES_DIR} not found")
        return
    detector = MTCNN()
    print("\n" + "="*70)
    print("IMAGE QUALITY CHECK - Analyzing Training Images")
    print("="*70)
    total_images = 0
    total_faces_detected = 0
    issues_found = []
    for person_name in os.listdir(FACE_IMAGES_DIR):
        person_dir = os.path.join(FACE_IMAGES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        print(f"\n📁 {person_name}:")
        print("-" * 70)
        person_images = 0
        person_faces = 0
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image_path = os.path.join(person_dir, filename)
            total_images += 1
            person_images += 1
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  ✗ {filename:30s} - Cannot read image")
                    issues_found.append(f"{person_name}/{filename}: Cannot read")
                    continue
                height, width = image.shape[:2]
                file_size = os.path.getsize(image_path) / 1024
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = detector.detect_faces(rgb_image)
                if not results:
                    print(f"  ✗ {filename:30s} - No face detected")
                    issues_found.append(f"{person_name}/{filename}: No face detected")
                elif len(results) > 1:
                    print(f"  ⚠ {filename:30s} - Multiple faces ({len(results)}) - Using first")
                    total_faces_detected += 1
                    person_faces += 1
                else:
                    face_box = results[0]['box']
                    face_width = face_box[2]
                    face_height = face_box[3]
                    face_area = face_width * face_height
                    image_area = width * height
                    face_percentage = (face_area / image_area) * 100
                    quality_issues = []
                    if face_percentage < 5:
                        quality_issues.append("Face too small")
                    if width < 200 or height < 200:
                        quality_issues.append("Low resolution")
                    if file_size < 20:
                        quality_issues.append("Low quality/compressed")
                    if quality_issues:
                        status = f"⚠ Warning: {', '.join(quality_issues)}"
                        issues_found.append(f"{person_name}/{filename}: {', '.join(quality_issues)}")
                    else:
                        status = "✓ Good"
                    print(f"  {status:40s} {filename:30s} ({width}x{height}, {face_percentage:.1f}% face)")
                    total_faces_detected += 1
                    person_faces += 1
            except Exception as e:
                print(f"  ✗ {filename:30s} - Error: {e}")
                issues_found.append(f"{person_name}/{filename}: Error - {e}")
        print(f"\n  Summary: {person_faces}/{person_images} images with valid faces")
        if person_images < 5:
            print(f"  💡 Recommendation: Add more images (current: {person_images}, recommended: 10-15)")
        if person_faces < person_images:
            print(f"  💡 Recommendation: Fix/replace images where no face was detected")
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"Total images: {total_images}")
    print(f"Images with faces detected: {total_faces_detected}")
    print(f"Success rate: {(total_faces_detected/total_images*100) if total_images > 0 else 0:.1f}%")
    if issues_found:
        print(f"\n⚠ Issues found: {len(issues_found)}")
        print("\nISSUES TO FIX:")
        for issue in issues_found[:10]:
            print(f"  • {issue}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found)-10} more")
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR BETTER ACCURACY:")
    print("="*70)
    print("1. Each person should have 10-15 high-quality images")
    print("2. Images should be at least 400x400 pixels")
    print("3. Face should occupy at least 20% of the image")
    print("4. Include varied conditions:")
    print("   - Different angles (front, slight left/right)")
    print("   - Different lighting (bright, normal, dim)")
    print("   - Different expressions (neutral, smiling)")
    print("   - With/without glasses (if applicable)")
    print("5. Avoid:")
    print("   - Blurry images")
    print("   - Multiple people in one image")
    print("   - Extreme angles or occlusions")
    print("="*70)


if __name__ == "__main__":
    check_image_quality()
