"""Versioned literature adapter inside the existing engineering workbench."""
from dataclasses import replace
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog

from grindcae380.desktop import FIELDS, BOUNDARIES, FORMULAS, default_case, case_to_form, form_to_case, load_case, save_case
from grindcae380.reconstruction import ReconstructionCase
from grindcae380.workflow import run_case, read_result

MODE = 'literature_grinding_force'


def read_history_published(directory):
    from grindcae.literature_history import read_result, MODE as history_mode
    from .analysis import analysis_mode_definition
    from .analysis_results import AnalysisPublishedResult, AnalysisResultError
    output=Path(directory).resolve()
    try:
        result=read_result(output)
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        raise AnalysisResultError('文献历史结果无法回读：'+str(exc)) from exc
    force=result['literature_force']; compare=result['comparison']
    lines=['任务：文献力驱动·弹塑性单程（均匀/条带）',
           f"重算稳态力：Fn={force['Fn_N']:.6g} N；Ft={force['Ft_N']:.6g} N；功率={force['power_W']:.6g} W",
           f"分项投影最大残差：{compare['maximum_component_projection_residual_N']:.3g} N",
           f"无支撑分量的均匀回退次数：{compare['uniform_fallback_component_count']}"]
    for name,label in [('baseline','均匀载荷'),('spatial','周向条带')]:
        final=result['routes'][name]['final_unloaded']
        lines.append(f"{label}：残余位移={final['maximum_residual_displacement_m']*1e6:.6g} μm；"
                     f"最大等效塑性应变={final['maximum_equivalent_plastic_strain']:.6g}；最终外载={final['external_force_norm_N']:.3g} N")
    lines.extend(['工程材料提供J2参数；β、断裂应力、摩擦、砂轮参数按文献页假设。',
                  '输出为固定网格宏观载荷响应；无真实材料分离、切屑或粗糙度。',
                  '3.8.1文献力与J2历史联算；独立实验验证未完成。'])
    definition=analysis_mode_definition(history_mode)
    return AnalysisPublishedResult(history_mode,definition.display_name,result['result_format'],output,
        output/'summary.json',output/'literature_history_comparison.png',result,'\n'.join(lines))


def configured_case(settings):
    return ReconstructionCase.from_mapping(settings.literature_case) if settings.literature_case is not None else default_case(calibrated=True)


def run_literature(case, output, progress=None):
    return run_case(case, output)


def read_published(directory):
    from .analysis import analysis_mode_definition
    from .analysis_results import AnalysisPublishedResult, AnalysisResultError
    output=Path(directory).resolve()
    try:result=read_result(output)
    except (ValueError,OSError) as exc:raise AnalysisResultError(str(exc)) from exc
    if result['result_format']!='grindcae_zhang2017_reconstruction_v2':
        raise AnalysisResultError('当前正式文献模式需要有界重建 Schema 2 结果。')
    c=result['normalized_input']['reference_case']
    text='\n'.join([
        '任务：文献磨削力·材料去除与塑性堆积',
        f"法向力 Fn={result['total_n_N']:.6g} N；切向力 Ft={result['total_t_N']:.6g} N",
        f"磨削功率 Ft·Vs={result['total_t_N']*c['wheel_speed_m_s']:.6g} W",
        f"动态磨粒 {result['counts']['dynamic']} / 总采样 {result['counts']['sampled']}",
        f"公式：{result['formula_convention']}；边界：{result['wheel_diagnostics']['boundary_rule']}",
        '四分量：塑性流动、断裂应力、前刀面摩擦、磨损面摩擦。',
        '当前输出为解析统计力模型；不包含FE场、真实材料分离或独立实验验证。',
        '文献切向比较差约10%–19%来自参考研究，不代表任意工况误差界限。',
        '完整参数、假设与校准来源见结果摘要和输入文件。',
    ])
    definition=analysis_mode_definition(MODE)
    return AnalysisPublishedResult(MODE,definition.display_name,result['result_format'],output,
        output/'summary.json',output/'force_components.png',result,text)


def sync_process(case,project):
    from .geometry import ParametricGeometryCase
    from .process_workspace import WorkbenchProcessCase
    geometry=ParametricGeometryCase.from_mapping(project.geometry)
    process=WorkbenchProcessCase.from_mapping(project.process)
    return replace(case,reference_case=replace(case.reference_case,
        wheel_diameter_m=geometry.wheel_diameter_m,depth_m=geometry.depth_of_cut_m,
        width_m=process.grinding_width_m,wheel_speed_m_s=process.wheel_surface_speed_m_per_s,
        feed_speed_m_s=process.workpiece_feed_speed_m_per_s),
        reconstruction_provenance=case.reconstruction_provenance+' Geometry/process explicitly copied from project '+project.project_name+'. Material/grit remain named literature inputs.')


class LiteratureParameterDialog:
    """Parameter-only modal editor; solving stays with the host AnalysisRunner."""
    def __init__(self,app):
        self.app=app
        self.window=tk.Toplevel(app.root);self.window.title('文献磨削力参数 · 随工程保存')
        self.window.geometry('900x710');self.window.transient(app.root)
        self.window.grab_set()
        outer=ttk.Frame(self.window,padding=12);outer.pack(fill='both',expand=True)
        toolbar=ttk.Frame(outer);toolbar.pack(fill='x')
        for text,command in [('参考算例',lambda:self.set_case(default_case(True))),('未标定算例',lambda:self.set_case(default_case())),
                             ('同步工程几何/工艺',self.sync),('加载文献JSON',self.load),('另存文献JSON',self.save)]:
            ttk.Button(toolbar,text=text,command=command).pack(side='left',padx=3)
        ttk.Label(outer,text='独立力模式使用本页参数。联算模式的轮径/切深/宽度/速度/屈服强度由工程覆盖；本页控制砂轮、β关系、断裂应力与摩擦假设。其他材料迁移未标定。',wraplength=830).pack(anchor='w',pady=9)
        container=ttk.Frame(outer);container.pack(fill='both',expand=True)
        canvas=tk.Canvas(container,highlightthickness=0);scroll=ttk.Scrollbar(container,command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');canvas.pack(fill='both',expand=True)
        form=ttk.Frame(canvas);window=canvas.create_window((0,0),window=form,anchor='nw')
        form.bind('<Configure>',lambda event:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda event:canvas.itemconfigure(window,width=event.width))
        self.variables={}
        for i,(name,label,_) in enumerate(FIELDS):
            row=i//2;col=i%2*2
            ttk.Label(form,text=label).grid(row=row,column=col,sticky='w',padx=4,pady=5)
            v=tk.StringVar();self.variables[name]=v
            ttk.Entry(form,textvariable=v,width=21).grid(row=row,column=col+1,sticky='ew',padx=4,pady=5)
        for row,(name,label,choices) in enumerate([('boundary_rule','边界',BOUNDARIES),('formula_convention','公式',FORMULAS)],start=8):
            ttk.Label(form,text=label).grid(row=row,column=0,sticky='w',padx=4,pady=5)
            v=tk.StringVar();self.variables[name]=v
            ttk.Combobox(form,textvariable=v,values=list(choices),state='readonly').grid(row=row,column=1,columnspan=3,sticky='ew',padx=4,pady=5)
        for row,(name,label) in enumerate([('wear_provenance','K₁来源'),('reconstruction_provenance','重建依据')],start=10):
            ttk.Label(form,text=label).grid(row=row,column=0,sticky='w',padx=4,pady=5)
            v=tk.StringVar();self.variables[name]=v
            ttk.Entry(form,textvariable=v).grid(row=row,column=1,columnspan=3,sticky='ew',padx=4,pady=5)
        form.columnconfigure(1,weight=1);form.columnconfigure(3,weight=1)
        self.status=tk.StringVar(value='应用后返回统一分析页计算；本窗口不单独运行求解器。')
        ttk.Label(outer,textvariable=self.status,wraplength=830).pack(fill='x',pady=8)
        ttk.Button(outer,text='应用到当前工程',command=self.apply).pack(side='right',padx=4)
        ttk.Button(outer,text='取消',command=self.window.destroy).pack(side='right',padx=4)
        self.set_case(configured_case(app._analysis_settings_from_widgets()))

    def set_case(self,case):
        self.base=case
        for key,value in case_to_form(case).items():
            choices=BOUNDARIES if key=='boundary_rule' else FORMULAS if key=='formula_convention' else {}
            self.variables[key].set(next((k for k,v in choices.items() if v==value),value))

    def current(self):
        values={k:v.get() for k,v in self.variables.items()}
        values['boundary_rule']=BOUNDARIES.get(values['boundary_rule'],values['boundary_rule'])
        values['formula_convention']=FORMULAS.get(values['formula_convention'],values['formula_convention'])
        return form_to_case(values,self.base)

    def apply(self):
        try:case=self.current()
        except (ValueError,OverflowError) as exc:self.status.set(str(exc));return
        self.app._literature_case_payload=case.to_dict()
        self.app._save_analysis_configuration()
        self.app._refresh_analysis_workspace()
        self.window.destroy()

    def sync(self):
        try:self.set_case(sync_process(self.current(),self.app.workbench.project));self.status.set('工艺已复制；请检查，应用后随工程保存。')
        except (ValueError,TypeError,KeyError) as exc:self.status.set('工程几何/工艺尚不完整：'+str(exc))

    def load(self):
        path=filedialog.askopenfilename(parent=self.window,filetypes=[('文献算例','*.json')])
        if path:
            try:self.set_case(load_case(path))
            except (ValueError,OSError) as exc:self.status.set(str(exc))

    def save(self):
        path=filedialog.asksaveasfilename(parent=self.window,defaultextension='.json',confirmoverwrite=False)
        if path:
            try:save_case(self.current(),path);self.status.set('已保存：'+path)
            except (ValueError,OSError) as exc:self.status.set(str(exc))
