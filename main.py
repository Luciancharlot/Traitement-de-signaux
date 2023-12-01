import cv2
import numpy as np

# Chemin vers votre fichier vidéo
video_path = './full.mp4'

# Création d'un objet de capture vidéo
cap = cv2.VideoCapture(video_path)

# Initialisation des compteurs
dans_le_magasin = 0
sortants = 0

# Position de la ligne de détection
ligne_de_detection_x_coord = 1200

# Initialisation du soustracteur de fond
fgbg = cv2.createBackgroundSubtractorMOG2()

# Seuils pour éviter les microvibrations
seuil_area = 30000  # Augmentation du seuil de l'aire
seuil_mouvement = 100 

# Identifiant unique pour chaque contour
next_id = 1
trackers = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Application du soustracteur de fond
    fgmask = fgbg.apply(frame)

    # Nettoyage de l'image de premier plan
    kernel = np.ones((5,5),np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Trouver les contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_contours = {}

    for contour in contours:
        if cv2.contourArea(contour) < seuil_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2

        same_object = None
        for oid, (ox, _) in trackers.items():
            if abs(ox - cx) < seuil_mouvement:
                same_object = oid
                break

        if same_object is not None:
            current_contours[same_object] = (cx, w)
        else:
            current_contours[next_id] = (cx, w)
            next_id += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    new_trackers = {}
    for oid, (cx, w) in current_contours.items():
        if oid in trackers:
            ox, _ = trackers[oid]
            if ox < ligne_de_detection_x_coord and cx >= ligne_de_detection_x_coord:
                if cx > ox:  
                    dans_le_magasin += 1
            elif ox > ligne_de_detection_x_coord and cx <= ligne_de_detection_x_coord:
                if cx < ox:
                    sortants += 1
                    dans_le_magasin = max(0, dans_le_magasin - 1)
        new_trackers[oid] = (cx, w)
    trackers = new_trackers

    cv2.line(frame, (ligne_de_detection_x_coord, 0), (ligne_de_detection_x_coord, frame.shape[0]), (255, 0, 0), 2)

    cv2.putText(frame, f'Dans le magasin: {max(dans_le_magasin, 0)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Sortants: {sortants}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground', fgmask)

    key = cv2.waitKey(60) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
