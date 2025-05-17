import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_WARN_ALWAYS'] = '0'

import numpy as np
import tensorflow as tf
import logging
import cv2
from model import VideoClassifier
from tqdm import tqdm
import pickle

# GPU ayarları ve kontrol
print("\n=== GPU Kontrolü ===")
print(f"TensorFlow sürümü: {tf.__version__}")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU bulundu: {len(physical_devices)} adet")
        print(f"GPU cihazları: {physical_devices}")
    except RuntimeError as e:
        print(f"GPU ayarları yapılandırılamadı: {e}")
else:
    print("GPU bulunamadı! CPU kullanılacak.")
print("===================\n")

# TensorFlow uyarı mesajlarını bastır
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def process_video(video_path, fps=30):
    """Video dosyasını frame'lere ayırır"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video açılamadı: {video_path}")
        return []
        
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo işleniyor: {os.path.basename(video_path)}")
    print(f"Toplam {total_frames} frame işlenecek...")
    
    with tqdm(total=total_frames, desc="Frame'ler işleniyor") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            pbar.update(1)
        
    cap.release()
    return frames

def resize_frames(frames, target_size=(224, 224)):
    """Frame'leri yeniden boyutlandırır"""
    return np.array([cv2.resize(frame, target_size) for frame in frames])

def process_videos_folder(folder_path):
    """Klasördeki tüm videoları işler"""
    if not os.path.exists(folder_path):
        print(f"Klasör bulunamadı: {folder_path}")
        return []
        
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    # Klasördeki tüm video dosyalarını bul
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(folder_path, file))
    
    if not video_files:
        print("Klasörde video dosyası bulunamadı!")
        return []
        
    print(f"\nToplam {len(video_files)} video dosyası bulundu.")
    all_frames = []
    
    # Her videoyu işle
    for video_path in video_files:
        frames = process_video(video_path)
        if frames:
            frames = resize_frames(frames)  # Frame'leri hemen boyutlandır
            all_frames.extend(frames)
            print(f"{len(frames)} frame işlendi.")
            
            # Her 1000 frame'de bir ara kayıt yap
            if len(all_frames) >= 1000:
                save_frames(all_frames, "processed_frames_partial.npy")
                all_frames = []  # Listeyi temizle
    
    # Kalan frame'leri kaydet
    if all_frames:
        save_frames(all_frames, "processed_frames_partial.npy")
    
    return True

def save_frames(frames, filename):
    """Frame'leri dosyaya kaydeder"""
    try:
        # Eğer dosya varsa, mevcut verileri yükle ve birleştir
        if os.path.exists(filename):
            existing_frames = np.load(filename, allow_pickle=True)
            frames = np.concatenate([existing_frames, frames])
        
        # Frame'leri kaydet
        np.save(filename, frames)
        print(f"Frame'ler kaydedildi: {filename}")
    except Exception as e:
        print(f"Frame'ler kaydedilirken hata oluştu: {str(e)}")

def load_frames_in_batches(filename, batch_size=1000):
    """Frame'leri parçalar halinde yükler"""
    try:
        # Dosya boyutunu kontrol et
        file_size = os.path.getsize(filename)
        if file_size == 0:
            print("Dosya boş!")
            return None
            
        # Frame'leri parçalar halinde yükle
        frames = np.load(filename, allow_pickle=True)
        total_frames = len(frames)
        
        print(f"Toplam {total_frames} frame yüklenecek")
        
        # Frame'leri parçalara böl
        for i in range(0, total_frames, batch_size):
            batch = frames[i:i + batch_size]
            yield batch
            
    except Exception as e:
        print(f"Frame'ler yüklenirken hata oluştu: {str(e)}")
        return None

def main():
    # Model oluştur
    classifier = VideoClassifier()
    
    while True:
        print("\n=== Video İşleme ve Model Eğitimi ===")
        print("1. Videolardan frame çıkar")
        print("2. Modeli eğit")
        print("3. Çıkış")
        
        choice = input("\nSeçiminiz (1-3): ")
        
        if choice == "1":
            try:
                folder_path = input("Video klasörünün yolunu girin: ")
                process_videos_folder(folder_path)
                print("Frame'ler başarıyla işlendi ve kaydedildi.")
            except Exception as e:
                print(f"Hata oluştu: {str(e)}")
                
        elif choice == "2":
            try:
                # Frame'leri parçalar halinde yükle ve eğit
                for batch_idx, frames_batch in enumerate(load_frames_in_batches("processed_frames_partial.npy")):
                    print(f"\nBatch {batch_idx + 1} işleniyor...")
                    # Örnek etiketler (gerçek uygulamada kullanıcıdan alınmalı)
                    labels = np.zeros(len(frames_batch))
                    
                    # Modeli eğit
                    history = classifier.train(frames_batch, labels)
                    print(f"Batch {batch_idx + 1} eğitimi tamamlandı.")
                
                # En iyi modeli kaydet
                classifier.save("video_classifier.h5")
                print("Model eğitimi tamamlandı ve kaydedildi.")
            except Exception as e:
                print(f"Hata oluştu: {str(e)}")
                
        elif choice == "3":
            print("Program sonlandırılıyor...")
            break
            
        else:
            print("Geçersiz seçim!")

if __name__ == "__main__":
    main() 