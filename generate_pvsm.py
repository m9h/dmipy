import os

def generate_pvsm(xdmf_path, pvsm_path="view_simulation.pvsm"):
    """
    Visualization Agent: Generates a Paraview State file (.pvsm) that automatically 
    loads the XDMF and applies Volume Rendering with Log-Scaling.
    """
    
    abs_xdmf = os.path.abspath(xdmf_path)
    
    # XML Template for Paraview State
    # Key settings embedded:
    # 1. Representation = "Volume"
    # 2. ColorArrayName = "log_photon_density"
    # 3. ScalarOpacityFunction = 0.0 at -6.0, 1.0 at -0.5 (Transfer Function)
    

    # 2. ColorArrayName = "log_photon_density"
    # 3. ScalarOpacityFunction = 0.0 at -6.0, 1.0 at -0.5 (Transfer Function)
    
    # 2. ColorArrayName = "log_photon_density"
    # 3. ScalarOpacityFunction = 0.0 at -6.0, 1.0 at -0.5 (Transfer Function)
    
    pvsm_template = f"""<?xml version="1.0"?>
<ParaView>
  <ServerManagerState Version="5.11.0" version="5.11.0">
    <Proxy group="sources" type="Xdmf3ReaderT" id="4526" servers="1">


      <Property name="FileNames" id="4526.FileNames" number_of_elements="1">
        <Element index="0" value="{abs_xdmf}"/>
      </Property>
      <Property name="PointArrays" id="4526.PointArrays">
        <Domain name="array_list" id="4526.PointArrays.array_list">
          <String text="log_photon_density"/>
          <String text="photon_density_norm"/>
        </Domain>
      </Property>
    </Proxy>
    
    <Proxy group="representations" type="UnstructuredGridRepresentation" id="4700" servers="1">
      <Property name="Input" id="4700.Input" number_of_elements="1">
        <Proxy value="4526"/>
      </Property>
      <Property name="Representation" id="4700.Representation" number_of_elements="1">
        <Element index="0" value="Volume"/>
      </Property>
      <Property name="Visibility" id="4700.Visibility" number_of_elements="1">
        <Element index="0" value="1"/>
      </Property>
      <Property name="ColorArrayName" id="4700.ColorArrayName" number_of_elements="1">

        <Element index="0" value="log_photon_density"/>
      </Property>
      <Property name="ScalarOpacityFunction" id="4700.ScalarOpacityFunction" number_of_elements="1">
        <Proxy value="9000"/>
      </Property>
      <Property name="ColorTransferFunction" id="4700.ColorTransferFunction" number_of_elements="1">
        <Proxy value="9001"/>
      </Property>
    </Proxy>
    
    <Proxy group="piecewise_functions" type="PiecewiseFunction" id="9000" servers="1">
      <Property name="Points" id="9000.Points" number_of_elements="8">
        <Element index="0" value="-6.0"/>
        <Element index="1" value="0.0"/>
        <Element index="2" value="0.5"/>
        <Element index="3" value="0.5"/>
        <Element index="4" value="-1.0"/>
        <Element index="5" value="0.8"/>
        <Element index="6" value="0.5"/>
        <Element index="7" value="0.5"/>
      </Property>
    </Proxy>
    
    <Proxy group="lookup_tables" type="PVLookupTable" id="9001" servers="1">
      <Property name="RGBPoints" id="9001.RGBPoints" number_of_elements="12">
        <Element index="0" value="-6.0"/>
        <Element index="1" value="0.0"/>
        <Element index="2" value="0.0"/>
        <Element index="3" value="0.0"/>
        
        <Element index="4" value="-2.0"/>
        <Element index="5" value="0.9"/>
        <Element index="6" value="0.0"/>
        <Element index="7" value="0.0"/>
        
        <Element index="8" value="0.0"/>
        <Element index="9" value="1.0"/>
        <Element index="10" value="1.0"/>
        <Element index="11" value="0.0"/>
      </Property>
    </Proxy>
    
    <Proxy group="views" type="RenderView" id="5000" servers="1">
      <Property name="Representations" id="5000.Representations" number_of_elements="1">
        <Proxy value="4700"/>
      </Property>
    </Proxy>
    
  </ServerManagerState>
</ParaView>
"""
    with open(pvsm_path, "w") as f:
        f.write(pvsm_template)
    
    print(f"Visualization Agent: Generated '{pvsm_path}' pointing to '{abs_xdmf}'.")

if __name__ == "__main__":
    generate_pvsm("kernel_flow_simulation.xdmf")
