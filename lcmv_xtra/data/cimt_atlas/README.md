# CIMT Unified Atlas
### Atlas parcellation and network selection

Source time courses were extracted for all regions of the **CIMT Unified Atlas**, a purpose-built 448-region-of-interest (ROI) parcellation assembled from three complementary sources to achieve comprehensive coverage of the circuits relevant to PD motor dysfunction.

<br>

- **Glasser + Tian Atlas (414 ROIs).** The Human Connectome Project multimodal cortical parcellation was combined with the Tian subcortical atlas, providing fine-grained coverage of cortical areas and standard subcortical nuclei including caudate, putamen, globus pallidus, thalamus, and nucleus accumbens.

- **Nettekoven Cerebellar Atlas (32 ROIs).** A symmetric cerebellar parcellation was incorporated to capture cerebellar contributions to motor timing and bimanual coordination, which are well-documented but typically absent from cortical-only atlases.

- **Custom STN Extraction (2 ROIs).** The left and right STN were explicitly represented using coordinate-based region extraction (5-mm radius spheres) centred on established MNI coordinates. This extraction was essential because the STN, the primary therapeutic target for DBS in PD, is insufficiently resolved in standard cortical atlases due to its small volume and deep location.

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

