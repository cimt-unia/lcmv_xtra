# CIMT Unified Atlas
### Atlas parcellation and network selection

Source time courses were extracted for all regions of the **CIMT Unified Atlas**, a purpose-built 448-region-of-interest (ROI) parcellation assembled from three complementary sources to achieve comprehensive coverage of the circuits relevant to PD motor dysfunction.

<br>

- **Glasser + Tian Atlas (414 ROIs).** The Human Connectome Project multimodal cortical parcellation was combined with the Tian subcortical atlas, providing fine-grained coverage of cortical areas and standard subcortical nuclei including caudate, putamen, globus pallidus, thalamus, and nucleus accumbens.

- **Nettekoven Cerebellar Atlas (32 ROIs).** A symmetric cerebellar parcellation was incorporated to capture cerebellar contributions to motor timing and bimanual coordination, which are well-documented but typically absent from cortical-only atlases.

- **Custom STN Extraction (2 ROIs).** The left and right STN were explicitly represented using coordinate-based region extraction (5-mm radius spheres) centred on established MNI coordinates. This extraction was essential because the STN, the primary therapeutic target for DBS in PD, is insufficiently resolved in standard cortical atlases due to its small volume and deep location.
- 
<br>

## Atlas Profile (448 ROIs)

### Structure
- **6 columns:** `index`, `roi_name`, `hemisphere`, `region_full_name`, `functional_system`, `sub_system`
- **Indices:** Sequential 0–447, no duplicates, no gaps
- **Hemisphere:** Perfectly balanced — 224 Left, 224 Right

### Functional Systems (18 unique)

| System | Count | Notes |
| :--- | :--- | :--- |
| Limbic | 64 | Largest system (hippocampus, amygdala, OFC, parahippocampal, striatum subfields) |
| DefaultMode | 64 | Precuneus, PCC, mPFC, angular gyrus, RSC |
| Visual | 54 | V1–V8, MT, FST, VMV, LO, dorsal/ventral streams |
| CinguloOpercular | 44 | Insula, frontal operculum, anterior/mid cingulate |
| DorsalAttention | 30 | IPS, FEF, PEF, SPL areas |
| Motor | 26 | M1, premotor, SMA, cerebellar motor regions |
| VentralAttention | 22 | TPJ, STS, AIP, temporal pole |
| Frontoparietal | 22 | DLPFC, IFJ, IFS, VLPFC |
| BasalGanglia | 22 | Putamen, caudate, globus pallidus, STN |
| Somatosensory | 20 | S1 (3a/3b/1/2), area 5, operculum |
| Thalamus | 16 | VA, VP, DA, DP/pulvinar subdivisions |
| Auditory | 14 | A1, belt, parabelt, association |
| Temporal | 14 | TGd, TGv, TE1/TE2, STS |
| Language | 10 | Broca, PSL, SFL, parainsular |
| Social | 10 | Cerebellar social regions only |
| Demand | 8 | Cerebellar demand regions only |
| Action | 6 | Cerebellar action regions only |
| Other | 2 | Gustatory cortex (R/L_43_ROI) |

### Sub-Systems (49 unique)
Key ones for selection logic:
- **Motor-relevant:** `Primary` (10), `Premotor` (14), `Supplementary` (4), `Eye` (2), `Cerebellum` (32), `Striatum` (20), `Pallidum` (4), `Subthalamic` (2), `DLPFC` (4), `IFJ` (4), `IFS` (4), `VLPFC` (4)
- **Basal ganglia sub-regions:** `Striatum`, `Pallidum`, `Subthalamic` (NOT labeled as "BasalGanglia" in sub_system — they use anatomical names)
- **Thalamic:** All 16 thalamic ROIs have `sub_system == "Thalamus"`


<br>

### References
1. Glasser, M.F. et al. A multi-modal parcellation of human cerebral cortex. Nature 536, 171–178 (2016).
2. Tian, Y., Margulies, D.S., Breakspear, M. & Zalesky, A. Topographic organization of the human subcortex unveiled with functional connectivity gradients. Nat. Neurosci. 23, 1421–1432 (2020).
3. Nettekoven, C. et al. A hierarchical atlas of the human cerebellum for functional precision mapping. bioRxiv (2024).

<br>

#### Additional Options

**Motor/Executive network selection for connectivity.** Rather than computing connectivity across the full 448×448 ROI space and subsequently selecting a subnetwork, we identified the Motor-Basal-Executive-STN ROI indices from the atlas metadata *before* any spectral estimation. The source epoch data, shaped as (n_epochs × 448 × n_times), were indexed to extract only the time courses belonging to the target network, producing a reduced array of shape (n_epochs × ~80 × n_times). All connectivity computations were then performed exclusively on this reduced dataset. This approach is computationally efficient, avoids inflating the multiple comparisons burden with irrelevant edges, and ensures that the resulting connectivity matrices are exactly 80×80 in dimension. The target network comprised approximately 80 ROIs spanning four functional systems:

- **Motor system**: primary motor cortex, premotor cortex, supplementary motor area (SMA), and frontal eye fields
- **Basal ganglia**: caudate, putamen, globus pallidus, nucleus accumbens, and bilateral STN
- **Executive/frontoparietal**: dorsolateral prefrontal cortex (DLPFC), inferior frontal junction (IFJ), inferior frontal sulcus (IFS), and ventrolateral prefrontal cortex (VLPFC)
- **Cerebellum**: motor and action-observation lobules

