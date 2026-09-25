from dataclasses import replace
from grindcae.gui.geometry import GeometryDisplayInput, Parametric2DGeometryProvider
from grindcae.gui.material_workspace import MaterialDisplayInput
from grindcae.gui.mesh_workspace import MeshDisplayInput
from grindcae.gui.process_workspace import ProcessDisplayInput, WheelDisplayInput
from grindcae.gui.project import ProjectNodeStatus, create_project
from grindcae.gui.analysis import AnalysisInputBuilder, AnalysisSettings
from grindcae380.desktop import default_case
from grindcae.literature_history import MODE


def project_and_settings(direction='positive_x', yield_mpa='225'):
    project=create_project('3.8.1 文献联算验收')
    geometry=Parametric2DGeometryProvider(GeometryDisplayInput(
        workpiece_length_mm='6',workpiece_height_mm='2',wheel_diameter_mm='300',
        depth_of_cut_um='15',wheel_lowest_point_x_mm='3',relative_feed_direction=direction).build_case()).to_geometry_schema()
    mesh=MeshDisplayInput(target_size_mm='.5',minimum_size_mm='.2',maximum_size_mm='1').build_case().to_dict()
    mesh['status']='completed'
    material=MaterialDisplayInput(label='440C文献屈服 + 演示弹性/硬化参数',yield_strength_mpa=yield_mpa,
        source_description='Yield 225 MPa from Zhang2017 unless explicitly changed for a synthetic plastic exercise; E=210 GPa, nu=0.3, Et=2 GPa are uncalibrated demonstration inputs.').build_case().to_dict()
    wheel=WheelDisplayInput(resolution_path='GB_FEPA_F',grit_designation='F80').build_case()
    process=ProcessDisplayInput(wheel_surface_speed_m_per_s='20',workpiece_feed_speed_m_per_s=str(1/30),
        grinding_width_mm='50').build_case(geometry,wheel).to_dict()
    project.geometry=geometry; project.mesh=mesh; project.material=material
    project.wheel=wheel.to_dict(); project.process=process
    for node in ('几何','网格','材料','工艺'):
        project.set_node_status(node,ProjectNodeStatus.READY)
        project.set_node_status(node,ProjectNodeStatus.COMPLETED)
    settings=replace(AnalysisSettings.recommended(),scan_base_position_count=5,literature_case=default_case(True).to_dict())
    return project,settings


def built_case(direction='positive_x',yield_mpa='225'):
    project,settings=project_and_settings(direction,yield_mpa)
    return AnalysisInputBuilder().build(project,MODE,settings).case
