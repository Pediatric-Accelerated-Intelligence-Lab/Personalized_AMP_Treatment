**Personalized AMP Treatment Programs**

GLM: Geometric Landmarking Module_

A pipeline that trains and applies a GDL model to place a fixed set of landmarks on 3D surface meshes. Given a mesh and a set of landmark definitions, it renders the surface from many random viewpoints, regresses per-landmark heatmaps in 2D, and back-projects the peaks into 3D by intersecting the view lines. Created for the purpose of annotating cleft lip and palate 3D meshes.

Based on the method described in:

    Paulsen, R. R., Juhl, K. A., Haspang, T. M., Hansen, T., Ganz, M., & Einarsson, G. (2018). Multi-view Consensus CNN for 3D Facial Landmark Placement. In Asian Conference on Computer Vision (pp. 706–719). Springer.

CPM: Clinical Prediction Module_

A pipeline that trains and applies a clinical outcome predictor on 3D landmark data. Given paired pre- and post-treatment landmark configurations per patient plus clinical variables, it builds a geometric feature vector set of pairwise distances, trains a regularized MLP, and reports both a per-patient probability and a ranked list of which features drive the prediction.

No external method required, CPM consumes 3D landmark text files directly.

Team
Artur Aharonyan
HyeRan Choo
Syed Anwar
