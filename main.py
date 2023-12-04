import cv2
import numpy as np

# Chemin vers votre fichier vidéo
video_path = './full.mp4'

# Création d'un objet de capture vidéo
cap = cv2.VideoCapture(video_path)

# Initialisation des compteurs
dans_le_magasin = 0
sortants = 0
entrants = 0  # Compteur pour les entrants

# Position de la ligne de détection
ligne_de_detection_x_coord = 1300

# Initialisation du soustracteur de fond
fgbg = cv2.createBackgroundSubtractorMOG2()

# Seuils pour éviter les microvibrations
seuil_area = 35000  # Seuil de l'aire
seuil_mouvement = 170  # Ajusté pour réduire les fausses détections

# Identifiant unique pour chaque contour
next_id = 1
trackers = {}

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_color_feature(contour, img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)  # Création d'un masque de type uint8
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(img, mask=mask)
    return mean_val[:3]  # Retourner seulement les composantes de couleur

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Application du soustracteur de fond
    fgmask = fgbg.apply(frame)

    # Trouver les contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_contours = {}

    for contour in contours:
        if cv2.contourArea(contour) < seuil_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2

        # Calcul des caractéristiques du contour
        color_feature = get_color_feature(contour, frame)

        # Logique améliorée de suivi des objets
        same_object = None
        for oid, (ox, oy, _, old_color) in trackers.items():
            if distance((ox, oy), (cx, cy)) < seuil_mouvement and distance(color_feature, old_color) < 50:  # Seuil de couleur à ajuster
                same_object = oid
                break

        if same_object is not None:
            current_contours[same_object] = (cx, cy, w, color_feature)
        else:
            current_contours[next_id] = (cx, cy, w, color_feature)
            next_id += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    new_trackers = {}
    for oid, (cx, cy, w, color_feature) in current_contours.items():
        if oid in trackers:
            ox, oy, _, _ = trackers[oid]
            if ox < ligne_de_detection_x_coord and cx >= ligne_de_detection_x_coord:
                if cx > ox:  
                    dans_le_magasin += 1
                    entrants += 1  # Incrémenter le compteur d'entrants
            elif ox > ligne_de_detection_x_coord and cx <= ligne_de_detection_x_coord:
                if cx < ox:
                    sortants += 1
                    dans_le_magasin = max(0, dans_le_magasin - 1)
        new_trackers[oid] = (cx, cy, w, color_feature)
    trackers = new_trackers

    # Dessiner la ligne de détection
    cv2.line(frame, (ligne_de_detection_x_coord, 0), (ligne_de_detection_x_coord, frame.shape[0]), (255, 0, 0), 2)

    # Afficher les compteurs
    cv2.putText(frame, f'Entrants: {entrants}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Dans le magasin: {max(dans_le_magasin, 0)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Sortants: {sortants}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground', fgmask)

    key = cv2.waitKey(70) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
