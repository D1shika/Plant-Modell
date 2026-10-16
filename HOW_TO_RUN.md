# How to run the leaf-health model yourself

Everything here was tested on this PC (Windows 11, Python 3.11 environment in the `.venv` folder). You do **not** need a GPU,
internet access, administrator rights, or to install anything: the trained models are already in the `models` folder.

## 1. Easiest way: drag and drop
1. Open the folder `C:\Plant model\fruit_health` in File Explorer.
2. Drag a leaf photo (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`) - or a whole folder of photos - onto **`run_predict.bat`**.
3. A black window opens, prints the result for each photo, and says where the files were saved. Press any key to close it.

## 2. From PowerShell (same thing, more options)
Open PowerShell inside the project folder: in File Explorer open `C:\Plant model\fruit_health`, click the address bar, type
`powershell` and press Enter. Then:

```powershell
# one photo
.\.venv\Scripts\python.exe src\predict.py --image "C:\Users\you\Pictures\leaf.jpg"

# all photos in a folder (not sub-folders)
.\.venv\Scripts\python.exe src\predict.py --folder "C:\Users\you\Pictures\my_leaves"

# optional: give environment values (see note below)
.\.venv\Scripts\python.exe src\predict.py --image "C:\Users\you\Pictures\leaf.jpg" --temp 28 --rainfall 10 --humidity 40 --soil_moisture 20

# optional switches
#   --variant image    use the image-only model (default is fusion; they are equally accurate)
#   --no-cam           skip the heatmap picture (faster)
```
Always put the path in double quotes (needed when it contains spaces). The first call takes a few seconds while the model loads.
It runs on the GPU if one is available, otherwise on the CPU automatically.

## 2b. In VS Code
The project contains a `.vscode` folder (interpreter + run buttons), and the Python extension is already installed.
1. VS Code -> **File -> Open Folder...** -> choose `C:\Plant model\fruit_health` (that folder itself, not its parent) and click
   "Yes, I trust the authors". (Or in PowerShell: `code "C:\Plant model\fruit_health"`.)
2. Check the interpreter: the bottom-right corner should show `3.11.0 ('.venv')`. If not: **Ctrl+Shift+P** -> *Python: Select
   Interpreter* -> pick the one inside `.venv` (or *Enter interpreter path...* -> `C:\Plant model\fruit_health\.venv\Scripts\python.exe`).
   If VS Code offers to "create an environment", say no.
3. Press **Ctrl+Shift+D** (Run and Debug) and pick an entry in the drop-down at the top, then press **Ctrl+F5** (Run Without
   Debugging; plain F5 also works but pauses on errors):
   - **"1 - Predict: SAMPLE photo (no questions)"** - first test, nothing to type; should print Grape / Healthy.
   - **"2 - Predict: the photo open in the editor"** - your own photo with no typing: open the photo in VS Code first
     (File -> Open File..., or drag it onto the VS Code window) so its tab is the active one, then run this entry.
   - **"3 - Predict: type the path of a photo"** - a box appears at the top centre of the window; replace the sample path with
     your own (no quotes) and press Enter. (An empty box makes the run fail with `--image: expected one argument`.)
   - 4 / 5: the same for a whole folder; 6: photo + environment values; 7: the 18-check self-test.
   The answer appears in the Terminal panel at the bottom.
4. Or use the terminal: **Ctrl+\`** opens one in the project folder; type
   `.\.venv\Scripts\python.exe src\predict.py --image "C:\path\to\leaf.jpg"`.
5. Results: Explorer (left) -> `outputs` -> `predictions` -> the newest folder -> click a `cam_...png` for the heatmap;
   `predictions.csv` opens as text (right-click -> *Reveal in File Explorer* to open it in Excel).

Notes: do not use the plain "Run Python File" button on `predict.py` (it needs an image argument and prints a usage error).
The terminal intentionally does not show `(.venv)`: auto-activation is switched off in `.vscode\settings.json` to avoid the
PowerShell "running scripts is disabled" error, so always call `.\.venv\Scripts\python.exe` as shown. The settings also keep
`.venv`, `data` and `models` (about 60,000 files) out of VS Code's file watcher and search to keep it light.

## 3. Where the results go
Every run creates a new folder `outputs\predictions\<date_time>\` containing
- `predictions.csv` - one row per photo (open it in Excel): crop, health, cause, condition, severity, confidences, which
  environment values were used;
- `cam_<photo name>.png` - the **Grad-CAM heatmap**: the photo on the left, on the right the same photo with red/yellow marking
  the areas that most influenced the condition prediction.

## 4. How to read the answer
```
Crop:       Apple (confidence 0.87)
Health:     Unhealthy (confidence 0.94)
Cause:      Biotic [only class seen in training ...]
Condition:  Apple___Black_rot (confidence 0.84)
Severity:   Mild (confidence 0.71) [heuristic damage estimate]
```
- **Crop** - one of Tomato, Apple, Grape, Strawberry, Peach, Cherry, Orange, Pepper_bell.
- **Health / Condition** - Healthy or Unhealthy, and the specific disease or pest (`Crop___Condition`; `Crop___healthy` for a healthy leaf).
- **Cause** - always shows Biotic for unhealthy leaves. The model had **no examples of nutrient/drought/waterlogging stress**, so it
  cannot recognise those. Do not read this line as a diagnosis.
- **Severity** - Mild / Moderate / Severe from a simple image-colour rule (share of the leaf that is not green). It is a rough
  estimate, not an expert rating.
- **Confidence** - how strongly the model favours its answer (0 to 1). It is **not** a probability of being correct; wrong answers
  can be confident, especially on photos that do not look like the training photos.
- **"Low confidence - this may not be one of the 8 supported crops or may be a poor photo"** - printed when a confidence is below 0.5.
  Treat that answer as unreliable (other plant? blurry? several leaves? not a leaf?).
- **"Environment values not provided: using training averages"** - normal. The four environment values (temperature, rainfall,
  humidity, soil moisture) were *simulated* during training, and the models learned to ignore them, so giving them changes nothing.

## 5. Getting the best results from your photos
- One leaf (or a leaf close-up) filling most of the picture, in focus, in good light, preferably on a plain background.
- Only the 8 crops above. Other plants, fruits, or whole-tree scenes give meaningless answers.
- iPhone `.HEIC` photos are not supported - save/convert them as `.jpg` first.
- Files must be under 50 MB and at least 32 x 32 pixels; unreadable or fake image files are rejected with a clear message.

## 6. How accurate is it? (measured on photos the model never saw)
- Clean lab-style photos (PlantVillage): condition correct about 99% of the time.
- Real field photos (PlantDoc): healthy/unhealthy correct about 95%, but the exact condition only about 63% (about 54% for the
  disease-only comparison). In a 3-photo demo of field images, 1 of 3 was fully correct.
- Cannot detect nutrient or water stress. Not for treatment or agronomic decisions. Details: `outputs\LIMITATIONS.md`.

## 7. If something goes wrong
| Message | What to do |
|---|---|
| `ERROR: no valid images to process` / `SKIPPED ...: file not found` | check the path and quotes; the file must be an image |
| `not a readable image` / `image too small` | the file is damaged, not really an image, or under 32 px |
| `--humidity=250 is outside the plausible range` | environment values must be sensible (humidity/soil 0-100, rainfall 0-2000, temp -50..70) |
| `No best.pt for variant ...` | the `models` folder is incomplete - restore it from your copy |
| `.venv\Scripts\python.exe not found` | keep `run_predict.bat` inside the `fruit_health` folder; see README.md to rebuild the environment |
| window closes instantly | run it from PowerShell (section 2) so you can read the message |

## 8. Other things you can run (optional)
```powershell
.\.venv\Scripts\python.exe -I src\evaluate.py          # re-compute all test tables in outputs\tables (about 5 min)
.\.venv\Scripts\python.exe -I src\train.py --all       # whole pipeline; skips every stage that is already finished
.\.venv\Scripts\python.exe -I src\smoke_test.py        # 18 quick self-checks of the model code
```
Re-training takes about 12 minutes per image model on this GPU (all four models: roughly 40-60 minutes; much longer on a CPU).
To retrain a model, first *move* its folder in `models` aside (do not delete it), then run
`.\.venv\Scripts\python.exe -I src\train.py --variant <image|env|fusion|fusion_shuffled_env>`.
Never overwrite `models\*\best.pt` without a backup.

## 9. Using it on another computer
Copy the whole `fruit_health` folder **except** `.venv` and `data\raw`, install Python 3.11, then follow "Environment setup"
in `README.md` to create a new `.venv`. For prediction only you need: `src\`, `models\` (the `best.pt` files, `label_maps.json`,
`env_stats.json`) and `requirements.txt`. The environment folder `.venv` does not work if simply copied to a different PC.
