# plan2cad

Scan de plan (A0…) → **DXF multi-calques calés** → **3D optionnelle**, en une commande, 100 % Python, sans étape manuelle.

| Étape du mode opératoire | Avant (manuel) | Maintenant (automatique) |
|---|---|---|
| Nettoyage du scan | GIMP : seuillage, retrait des taches | aplatissement du fond, seuil d'Otsu, suppression des poussières / textes |
| Vectorisation | VTracer / WinTopo (lignes centrales) | squelette + graphe + fusion des tronçons alignés + Douglas-Peucker |
| Calques & couleurs | à la main dans QCAD | charte `ARC-… / ELEC-… / PLUMB-… / CVC-…`, couleurs forcées |
| Calage (ALIGNER) | 2 points à cliquer | Fourier-Mellin + ICP automatique, ou 2 points d'ancrage dans un fichier |
| Repères pour l'architecte | à dessiner | calque `Z-POINTS-ANCRAGE` : 2 croix, générées |
| Passage en 3D | extrusion dans FreeCAD | `--3d` : OBJ, STL, DXF 3DFACE, visionneuse web |

## Installation

```bash
cd plan2cad
pip install .            # ou : pipx install .   (commande `plan2cad` disponible partout)
pip install pymupdf      # optionnel : lecture PDF sans poppler
```
Dépendances : numpy, opencv-python-headless, scikit-image, scipy, pillow. Les PDF sont lus par `pdftoppm` (poppler-utils) ou PyMuPDF. Le DXF est écrit/lu par un module interne : pas de dépendance à `ezdxf`.

## Essayer tout de suite

```bash
plan2cad demo -o essai --3d --serve      # scans fictifs bruités, chaîne complète, interface web
```

## Utilisation

```bash
# chaîne complète : un scan par lot, calés sur l'architecture
plan2cad run --arc archi.tif --elec elec.tif --plumb plomberie.pdf -o sortie --scale 100 --dpi 400

# avec la 3D (optionnelle) et l'interface web pendant le traitement
plan2cad run --arc archi.tif --elec elec.tif -o sortie --3d --serve

# traitement par lots : un DXF par scan
plan2cad vectorize scans/*.tif --lot arc -o dxf/ --remove-text

# 3D à partir d'un DXF 2D déjà existant
plan2cad extrude plan.dxf --wall-height 2700

# rouvrir l'interface web sur des résultats existants
plan2cad serve sortie/

plan2cad layers        # charte de calques
```
Entrées acceptées : PNG, TIFF (1 bit OK), JPEG, PDF (page 1), DXF.
Sorties dans le dossier : `<nom>.dxf` (tous lots), `<nom>_ARC.dxf`, `_ELEC.dxf`, `_PLUMB.dxf`…, `<nom>.svg`, `steps/*.png`, `run.json`, et avec `--3d` : `_3d.obj/.stl/.dxf`, `model3d.json`.

Unités : **mm réels**. `--scale 100` = plan au 1:100 (défaut). `--dpi` = résolution du scan (défaut 400).

### Interface web
`--serve` (ou `plan2cad serve DOSSIER`) ouvre <http://127.0.0.1:8765> :
* **Étapes** : chaque étape avec durée, statistiques et image (scan → fond corrigé → binaire → nettoyé → squelette → vecteurs), zoom plein écran ; mise à jour en direct pendant le calcul.
* **Plans** : le DXF final par calques (cases à cocher), zoom / déplacement.
* **3D** : vue orbitable, calques activables (onglet présent seulement avec `--3d`).
* **Fichiers** : téléchargements.

### Fichier projet (plusieurs niveaux, points d'ancrage manuels)
```json
{
  "name": "immeuble",
  "levels": [
    {"name": "RDC", "z": 0, "height": 2700,
     "files": {"arc": "rdc_archi.tif", "elec": "rdc_elec.tif"},
     "anchors": {"elec": {"src": [[812, 4410], [17630, 12020]],
                          "dst": [[640, 4390], [17420, 12010]]}}},
    {"name": "R+1", "z": 2700, "height": 2600,
     "files": {"arc": "r1_archi.tif", "elec": "r1_elec.tif"}}
  ]
}
```
`anchors` : coordonnées **en pixels de l'image d'origine** (deux points du scan du lot → les mêmes points sur le scan d'architecture). Sans `anchors`, le calage est automatique. Les niveaux sont calés entre eux automatiquement et empilés en 3D.

## Dossier de plans quelconque (scan → manifeste → build)

Couleur ou noir & blanc, un dossier par bâtiment ou par lot, une planche = 1 plan ou 4-5 plans : donnez le tout.

```bash
plan2cad scan  mes_plans/ -o sortie        # inventaire + manifest.json (à relire, corriger)
plan2cad build sortie/manifest.json -o res # un dossier de résultats par bâtiment (DXF par niveau)
plan2cad auto  mes_plans/ -o sortie        # les deux d'un coup
plan2cad serve sortie                      # onglet « Inventaire » : arborescence, confiance, vignettes annotées
```

Ce que fait `scan` : retire le cadre de planche, découpe la planche en plans (les textes ne font pas pont), ignore
cartouches et doublons, lit bâtiment / niveau / lot dans les noms de fichiers et dossiers, sinon les déduit du dessin
(murs pleins ou double trait = architecture ; couleurs = lots ; plans N&B sans indice = lot « autre »), regroupe les plans
d'architecture à emprise identique en un bâtiment (niveaux = plans successifs), puis rattache chaque plan de lot à
l'architecture dont la géométrie coïncide. Chaque décision porte une **confiance** et ses **raisons** dans le manifeste ;
rien n'est silencieux. Modifiez `building`, `level`, `lot`, `skip` à la main puis relancez `build`.

Limites : un lot en N&B sans nom ni couleur reste « autre » ; deux bâtiments aux emprises quasi identiques et sans nom
peuvent être confondus (signalé par une confiance faible).

## Réglages utiles
| Option | Rôle |
|---|---|
| `--max-px 7000` | côté de travail max ; un A0 à 400 dpi (18 700 px) est réduit (≈ 17 mm réels / pixel à 1:100). Monter à 10000-12000 pour plus de précision (≈ 3 Go de RAM). |
| `--remove-text [lot…]` | supprime les petits éléments type texte (≤ `--text-mm`). Retire aussi les petits symboles : à réserver à l'architecture. |
| `--deskew` | redresse un scan penché (±3°). |
| `--threshold N` | seuil noir/blanc manuel si l'automatique rate des traits pâles. |
| `--simplify / --min-length / --prune` | tolérances (mm papier) de simplification, longueur min, épines. |
| `--strip-backdrop [MM]` | retire des lots le fond d'architecture qui recouvre le plan maître. |
| `--align none` | pas de calage. |
| `--3d --wall-height 2700 --wall-thickness 200 --elev elec=2400,plumb=300` | paramètres 3D. |

## Limites connues (à connaître avant de s'y fier)
* **Textes (OCR)** : lus avec Tesseract s'il est installé (`apt install tesseract-ocr tesseract-ocr-fra` ; sans lui, tout marche mais le texte n'est pas lu, message dans le journal). Méthode : l'outil isole lui-même les blocs de texte (lettres regroupées par taille et alignement), puis lit chaque bloc en horizontal, de bas en haut et de haut en bas, avec agrandissement automatique du petit texte (≥ 0,55 mm papier de haut). Il sert à : identifier le lot d'un plan sans nom (« ÉLECTRICITÉ », « PLOMBERIE », « CHAUFFAGE », « GAZ »…), lire le niveau (« PLAN NIVEAU 1 », « RDC », « R+2 »), reconnaître les noms de pièces. Les textes sont écrits en objets TEXTE (avec leur angle) sur `ARC-TEXTE` / `ELEC-TEXTE` / `PLUMB-TEXTE`…, les lectures douteuses (confiance < 70 %) sur `Z-TEXTE-INCERTAIN` ; les lettres sont retirées du tracé. **Écriture manuscrite : non lue** (Tesseract n'en est pas capable ; elle donne des lectures de faible confiance, classées « incertain », ou rien). Texte oblique (ni 0° ni 90°) : non lu. `--no-ocr` pour désactiver.
* **Cotes** : un nombre (« 8.26 », « 350 », « 3,50 m ») est relié à la ligne droite parallèle la plus proche, qui passe sur `ARC-COTES` avec son texte. Les longueurs écrites et dessinées donnent l'échelle réelle du plan : si ≥ 3 cotes concordent à ±6 %, l'outil **avertit quand elle diffère de `--scale`** (une mauvaise échelle fausse toutes les dimensions et la 3D), et `--scale auto` l'applique. Limites : une cote dont le texte coupe la ligne en deux, des chaînes de cotes sur une même ligne, ou des cotes d'autre nature (altimétrie, angles) ne sont pas exploitées ; la valeur n'est pas comparée trait par trait à la géométrie.
* **Portes, fenêtres, escaliers** (`--no-openings` pour désactiver). Ouverture = deux tronçons de mur alignés séparés de 0,5 à 3 m : porte si un arc de battant OU un battant seul part d'un montant, fenêtre si des traits fins la traversent, sinon baie ouverte (classée `ARC-PORTE`). **Double trait** : les fenêtres dessinées sans interrompre le mur (2-3 traits dans son épaisseur sur 0,4 à 3 m) sont reconnues et le mur est coupé à leur droit ; les escaliers ne sont plus noyés par le comblement des doubles traits (on ne comble que les bandes de largeur constante entre deux traits de mur). Escaliers : ≥ 4 traits parallèles de même longueur espacés de 17 à 42 cm, ou ≥ 5 marches **en éventail** (traits dont les prolongements passent par un même centre, régulièrement répartis en angle). Battants, arcs, traits de fenêtre et marches vont sur `ARC-PORTE` / `ARC-FENETRE` / `ARC-ESCALIER`, plus un repère en travers de chaque ouverture et l'emprise de chaque escalier. En 3D : porte = panneau + linteau, fenêtre = allège + vitrage + linteau, escalier non modélisé. **Non reconnus** : portes coulissantes sans symbole, portes doubles dessinées sans battant, fenêtres sans aucun trait intérieur, escaliers dont les marches ne sont pas des traits droits (marches courbes), escaliers dessinés par arcs concentriques seuls, symboles sur plans scannés très dégradés. Tout est reclassé, rien n'est supprimé (sauf le mur coupé au droit d'une fenêtre en double trait).
* Tout le reste du gros œuvre va dans `ARC-MURS-*` ; la séparation EXT / INT se fait par épaisseur de trait, seulement si la distribution est nettement bimodale, sinon tout est `ARC-MURS-INT`. Les lots réseaux n'ont qu'un calque par lot (`ELEC-COURANT-FORT`, `PLUMB-EAU-POTABLE`, `CVC-VENTIL`) : reventilation à faire ensuite.
* **Murs en double trait** : chaque trait est vectorisé séparément (on ne reconstitue pas la ligne centrale du mur). L'extrusion 3D donne alors des voiles minces le long de chaque trait. Elle est fidèle pour les murs pleins (poché noir). Les ouvertures détectées sont découpées (porte, fenêtre) uniquement pour les murs pleins.
* **Calage automatique** : il faut de la géométrie commune avec l'architecture (les murs en fond des plans de lots suffisent). Un plan de réseau nu, sans fond, se cale mal : un avertissement s'affiche (`inlier_ratio` bas), utiliser alors les points d'ancrage.
* **Fond d'architecture dupliqué** dans les lots : signalé (`recouvrement_plan_maitre`). `--strip-backdrop` le retire, mais retire aussi les réseaux posés le long des murs (tolérance 60 mm par défaut).
* Réseaux 3D : tracés à altitude constante, sans colonnes montantes.
* DXF produit en **R12 ASCII**, validé par un vérificateur structurel et une relecture interne ; **non essayé dans AutoCAD / FreeCAD / QCAD** lors du développement : ouvrir un premier fichier dans votre logiciel avant de l'industrialiser.

## Tests
```bash
python -m unittest discover -s tests -v
```

## Mises à jour récentes

- **Échelle** : `--scale 100`, `1:48`, `1/4"=1'` ou `auto` (déduite des cotes métriques/impériales, ou des dimensions de pièces lues). Si rien n'est lisible, `--known-length 17.4` / `57ft` / `57'6"` cale le plus grand côté du bâtiment puis s'aligne sur l'échelle standard si l'écart est < 3 %.
- **Ouvertures** : une brèche nue dans la façade devient une fenêtre (confiance 0,5, « à vérifier ») ; escaliers à arcs concentriques et en éventail détectés ; portes coulissantes/doubles sans symbole classées « ouverture ».
- **Calques** : `Z-TEXTE-NON-LU` (blobs de lettres illisibles), `Z-TEXTE-INCERTAIN` (OCR < 70).
- **Inventaire** : découpe XY-cut (bande blanche ≥ 7 mm), type « illustration » (croquis ignorés).
- **Limites réelles** (scan de catalogue ancien) : écriture manuscrite non lue ; croquis et plan sans bande blanche restent fusionnés (recadrer dans le manifest) ; escaliers à marches très courtes non détectés ; OCR pleine page lent (12–22 s).

## PDF vectoriels (export CAO) : lecture directe

Un PDF issu d'un logiciel de CAO n'est **pas rastérisé** : les tracés (épaisseur, couleur) et les textes (chaîne, position, angle) sont lus tels quels (`pip install pdfplumber`). Par rapport à un scan : géométrie exacte, textes sans erreur d'OCR, quelques secondes par planche.

- **Planche → plans** : amas de tracés ; chaque plan est titré par le texte voisin (« RDC EXISTANT », « R+1 PROJET 3 »). Niveau = RDC / R+n ; **variante (Existant, Projet 3) = nom du bâtiment**, donc les niveaux d'une même variante sont calés entre eux.
- **Calques** : épaisseur de trait (fort → `ARC-MURS-EXT`, moyen → `ARC-MURS-INT`, fin → `ARC-DETAIL`), arcs de battant → `ARC-PORTE`, lignes de cote → `ARC-COTES`, tracés colorés → `ARC-COULEUR-<couleur>` (les couleurs d'un plan d'architecture ne sont pas des réseaux ; pour un plan de lots en couleur, utiliser `--legend`).
- **Échelle** (`--scale auto`, désormais le défaut) : cotes en texte, sinon surfaces de pièces, sinon largeur des portes (≈ 0,90 m), sinon 1:100. Avertissement si elle diffère de `--scale`.
- **Cotes dessinées en tracés** (texte converti en courbes, fréquent en export CAO) : les chiffres sont reconnus par comparaison de formes (nombre de boucles + gabarits) et deviennent de vrais textes ; ils servent à l'échelle de **toute la planche**. Chiffres serrés ou superposés : parfois manqués ou mal lus (les cotes aberrantes sont écartées par le vote de concordance).
- **Murs** : les tracés de même épaisseur que les murs mais isolés (mobilier) sont écartés vers `ARC-DETAIL` ; deux traits parallèles proches (6 à 60 cm) donnent un **axe de mur avec épaisseur réelle** (`ARC-MURS-AXE`, extrudé en 3D) ; les tracés d'origine restent sur `ARC-MURS-*`. Seuls les murs horizontaux/verticaux sont appariés.
- **Ouvertures sans symbole** : trou aligné (0,55 à 3,2 m) dans un mur → `ARC-PORTE` s'il y a un arc de battant, `ARC-FENETRE` s'il est sur la façade (à vérifier), sinon passage (`ARC-PORTE`, marqueur). 
- `--raster` force l'ancien traitement par image.


## Scans : redressement et régularisation

Pour les plans scannés/photographiés, le squelette « tremble » et la feuille n'est jamais parfaitement droite. Automatiquement :
1. **Redressement (skew)** : l'angle dominant des murs (modulo 90°) est mesuré (0,1° de précision) et la feuille est tournée pour que les murs soient horizontaux/verticaux ; textes et cotes suivent. Actif dès 0,12° et jusqu'à 20°.
2. **Régularisation** : les suites de segments quasi H/V deviennent un seul segment exact, les murs alignés partagent la même coordonnée, les angles sont recalculés par intersection, les sommets inutiles disparaissent.
Les diagonales, arcs et façades obliques ne sont pas modifiés. Désactivation : `--no-regularize`. Les stats (`redressement_deg`, `murs_exacts_avant_apres`) sont dans `run.json`.

Extensions : les obliques droites (façades, pans coupés) sont redressées et alignées sur 30/45/60° ou sur l'orientation commune de plusieurs façades ; les arcs de porte et murs courbes bruités deviennent de vrais arcs ; les PDF vectoriels reçoivent le même redressement (rotation avant analyse) et un alignement à tolérance serrée ; les petits amas isolés (mobilier) passent en `ARC-DETAIL` aussi pour les scans.

## Lettrage à la main : planche de relecture

Les lettres manuscrites (capitales d'architecte) ne sont pas fiables à lire automatiquement sans modèle d'apprentissage (essais : gabarits + vocabulaire, Tesseract mot par mot : trop de faux). Plutôt que de deviner :
1. l'outil sort `<plan>_a_relire.png` (vignettes numérotées L001, L002…) et `<plan>_a_relire.json` ;
2. remplir le champ `texte` de chaque ligne dans le JSON ;
3. relancer avec `--texts <plan>_a_relire.json` : les textes sont posés à l'emplacement des lettres et les lettres disparaissent du dessin.
Les identifiants restent stables tant que les mêmes options sont utilisées.


## Arcs et cercles vrais (ajustement de primitives)

Les cercles et arcs nets (arcs de battant, murs courbes, poteaux ronds) sont ajustés sur les **pixels bruts du squelette**
(centaines de points : précision sous-pixel) puis écrits en vraies entités DXF `ARC` / `CIRCLE`, au lieu d'une polyligne à
facettes. Ils ne sont plus aplatis par la régularisation H/V. Un garde-fou refuse les faux positifs (cloisons droites bruitées,
coins de rectangle, tronçons trop courts ou trop plats).

Mesure sur un plan synthétique à géométrie connue (`python tests/bench_fidelity.py`) : écart moyen ≈ 0,08 mm papier inchangé,
précision 0,121 → 0,108 mm, sommets 100 → 44, entités `ARC`/`CIRCLE` 0 → 3 / 1. Tests : `python -m unittest tests.test_fit`.
Limites : arcs ajustés seulement sur les branches simples du squelette ; un arc fusionné avec un trait droit avant ajustement est
repris par un ajustement sur sommets (moins précis) ; pas encore ouvert dans AutoCAD / QCAD (seul le relecteur interne a validé).


## Métrés : murs -> pièces -> surfaces (`plan2cad metres`)

    python -m plan2cad metres plan.dxf -o metres/      # JSON, CSV, DXF des pièces (calque ARC-PIECES), PNG de contrôle

Les faces de murs (`ARC-MURS-INT`) sont tracées sur une grille, fermées (épaisseur pleine), les portes ferment les passages
(segments du calque `ARC-PORTE`), puis chaque espace libre fermé est une pièce : surface nette, périmètre, nombre de portes et de
fenêtres, libellé lu dans la pièce. Si le plan porte une surface écrite (« 5,27 m² »), elle est comparée à la surface calculée.

**Contrôle sur le plan SDIS fourni** (4 plans, PDF vectoriel) : les pièces sont bien isolées, mais sur le Projet 3 la surface calculée
dépasse la surface écrite de ≈ 17 % en moyenne (écart constant d'une pièce à l'autre : facteur linéaire 0,92, écart-type 1 à 2 %).
Cela ressemble à une autre définition de la surface écrite ou à une échelle différente de 8 %, pas à du bruit ; l'échelle lue sur les
cotes (1:229,6) est cohérente avec les largeurs de portes (0,83 / 0,90 / 0,93 m). À trancher avec une cote mesurée sur site.
Limites : un espace sans mur ni porte qui communique largement avec un autre forme une seule pièce ; le mobilier n'est pas déduit ;
les angles rentrants sont arrondis (≈ 7 cm de périmètre en moins par angle) ; lit `ARC-MURS-INT` (en mode scan brut, les murs et les
portes ne sont pas encore reconnus sur ce type de plan, donc les métrés n'y sont pas fiables).

## Tester le mode scan brut à partir d'un PDF vectoriel

    python tools/pdf_to_fake_scans.py planche.pdf scans/ --dpi 300
    python -m plan2cad vectorize scans/scan-4.png --dpi 300 --scale 229.55

Le PDF vectoriel sert de vérité terrain : `plan2cad auto` sur le PDF donne le DXF de référence.


## Mode scan : murs en double trait, portes, échelle

* **Murs en double trait** (`vectorize.detect_wall_gaps`) : on mesure directement l'écart entre traits parallèles LONGS
  (≥ 12 mm papier, horizontaux / verticaux). Chaque pic de l'histogramme est une épaisseur de mur (cloison ≈ 0,7 mm
  papier, mur porteur ≈ 1,8 mm). Le mobilier ne noie plus le signal (l'ancienne granulométrie du fond reste en repli).
  Le comblement ne remplit que les bandes entre deux traits longs et droits : arcs de porte, escaliers, chaises et
  texte ne sont jamais noircis.
* **Portes à battant** (`doors.py`) : gabarit « vantail + quart de cercle » corrélé avec l'encre sur 8 orientations
  et ~30 rayons ; conditions : vantail encré, arc encré, quart de disque balayé vide, anneaux voisins vides, mur de part
  et d'autre de l'ouverture. Le rayon = largeur de la porte, mesurée sans connaître l'échelle.
* **Échelle** : cotes > surfaces écrites (« 13 m² », y compris lues « m? » par l'OCR) > largeur des portes simples
  (≈ 0,85 m, **approximatif ±25 %**, appliqué seulement avec `--scale auto`). Sinon l'échelle est déclarée
  *non déterminée* (avertissement), plus de repli silencieux.
