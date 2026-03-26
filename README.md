# 2D-LBM (D2Q9)

A D2Q9 LBM code is presented which inputs the segmented AVI stacks (pore space is white) and calculates the single phase steady state permeability. The LBM code uses BGK and Guo forcing conditions and has no-slip bounce-back walls on the top and bottom boundaries. The body force is in the +X-direction. The output includes:
    - permeability_profile.csv                
    - K_over_Ki_vs_IVR.png                    
    - velocities_stats.csv                    
    - velocities_profiles.csv                 
    - velocity_fields.npz                     
    - velocity_maps/vel_slice_*.png          

