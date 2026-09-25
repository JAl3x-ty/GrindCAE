"""Chinese desktop workbench for the independent literature force kernel."""
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import uuid

from . import __version__
from .core import canonical
from .dispatch import read_case
from .reconstruction import ReconstructionCase
from .workflow import run_case, read_result

# SI units per displayed unit. Editing never infers units from magnitudes.
FIELDS = (
    ('wheel_diameter_m','砂轮直径 (mm)',1e-3),('width_m','磨削宽度 (mm)',1e-3),
    ('depth_m','磨削深度 (μm)',1e-6),('wheel_speed_m_s','砂轮速度 (m/s)',1.),
    ('feed_speed_m_s','进给速度 (m/min)',1/60),('friction_coefficient','摩擦系数',1.),
    ('yield_stress_Pa','屈服应力 (MPa)',1e6),('fracture_stress_Pa','断裂应力 (MPa)',1e6),
    ('cone_half_angle_rad','磨粒半角 (rad)',1.),('grain_min_m','粒径下限 (μm)',1e-6),
    ('grain_max_m','粒径上限 (μm)',1e-6),('organization_number','组织号',1.),
    ('seed','随机种子',1.),('relief_depth_m','修整退缩深度 (μm)',1e-6),
    ('wear_coefficient_N_m','磨损系数 K₁ (N·m)',1.))
BOUNDARIES={'有限入口':'finite_entry','周期代表区':'periodic_envelope'}
FORMULAS={'连续应力与摩擦投影':'continuous_projection','原文印刷公式':'printed'}


def default_case(calibrated=False):
    c=ReconstructionCase.paper_example()
    if calibrated:
        c=replace(c,reference_case=replace(c.reference_case,wear_coefficient_N_m=5.283606805687778,
            wear_provenance='Reference fit: paper Fig18 dry predicted Fn=65.53 N, seeds 2017..2032, '
            'finite_entry, relief=90 um, continuous_projection, Table6 conditions. '
            'Normal-only ensemble inverse; K1=5.283606805687778 N m; not independent experimental validation. '
            'Kept fixed when process is changed; altered geometry/boundary/material requires new calibration.'))
    return c


def case_to_form(case):
    if not isinstance(case,ReconstructionCase):raise ValueError('参数编辑支持有界重建算例；旧版结果可通过“打开结果”查看。')
    c=case.reference_case
    values={name:str(getattr(c,name)) if name in ('seed','organization_number') else format((case.relief_depth_m if name=='relief_depth_m' else getattr(c,name))/scale,'.12g')
            for name,_,scale in FIELDS}
    values.update(boundary_rule=case.boundary_rule,formula_convention=c.formula_convention,
                  wear_provenance=c.wear_provenance,reconstruction_provenance=case.reconstruction_provenance)
    return values


def form_to_case(values,base):
    kw={}
    original=case_to_form(base)
    relief=base.relief_depth_m
    for name,label,scale in FIELDS:
        try:
            value=int(values[name]) if name in ('seed','organization_number') else float(values[name])*scale
        except (ValueError,TypeError,OverflowError) as exc:raise ValueError(f'{label}：请输入有效数值。') from exc
        if values[name]==original[name]:
            value=base.relief_depth_m if name=='relief_depth_m' else getattr(base.reference_case,name)
        if name!='relief_depth_m':kw[name]=value
        else:relief=value
    kw.update(formula_convention=values['formula_convention'],wear_provenance=values['wear_provenance'])
    if kw['wear_coefficient_N_m']!=base.reference_case.wear_coefficient_N_m and kw['wear_provenance']==base.reference_case.wear_provenance:
        raise ValueError('修改 K₁ 后，请同时填写新的磨损系数来源。')
    return replace(base,reference_case=replace(base.reference_case,**kw),
                   relief_depth_m=relief,boundary_rule=values['boundary_rule'],
                   reconstruction_provenance=values['reconstruction_provenance'])


def load_case(path):
    return read_case(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def save_case(case,path):
    # Exclusive creation: never replace a user's earlier case.
    text=json.dumps(case.to_dict(),indent=2,ensure_ascii=False,allow_nan=False)
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(text)


class Workbench:
    def __init__(self,root,output_root=None):
        self.root=root
        self.output_root=Path(output_root or Path(__file__).resolve().parents[2]/'outputs'/'desktop_runs')
        self.busy=False;self.last_error=None;self.result=None;self.result_directory=None
        self.pending_output=None;self.result_state='none';self._setting=False;self.dirty=False
        self._events=queue.Queue();self._photo=None;self.started=0.
        self.root.title(f'GrindCAE {__version__} · 文献力模型工作台')
        sw,sh=root.winfo_screenwidth(),root.winfo_screenheight()
        root.geometry(f'{min(1280,sw-70)}x{min(860,sh-100)}')
        root.minsize(850,560)
        style=ttk.Style(root);style.theme_use('clam')
        style.configure('.',font=('Microsoft YaHei UI',10))
        style.configure('Title.TLabel',font=('Microsoft YaHei UI',17,'bold'))
        self.frame=ttk.Frame(root,padding=14);self.frame.pack(fill='both',expand=True)
        ttk.Label(self.frame,text='文献磨削力工作台',style='Title.TLabel').pack(anchor='w')
        ttk.Label(self.frame,text='Zhang 2017 · 材料去除 / 塑性流动 / 摩擦分项 · 独立并行版本').pack(anchor='w',pady=(3,10))
        toolbar=ttk.Frame(self.frame);toolbar.pack(fill='x')
        self.controls=[]
        for label,callback in [('参考标定算例',lambda:self.choose_default(True)),('未标定算例',lambda:self.choose_default(False)),
                               ('加载算例',self.load_dialog),('另存算例',self.save_dialog),('打开结果',self.open_dialog)]:
            b=ttk.Button(toolbar,text=label,command=callback);b.pack(side='left',padx=(0,6));self.controls.append(b)
        self.run_button=ttk.Button(toolbar,text='开始计算',command=self.start_calculation);self.run_button.pack(side='right')
        self.notebook=ttk.Notebook(self.frame);self.notebook.pack(fill='both',expand=True,pady=12)
        parameter_page=ttk.Frame(self.notebook)
        self.parameter_canvas=tk.Canvas(parameter_page,highlightthickness=0,background='#f0f0f0')
        parameter_scroll=ttk.Scrollbar(parameter_page,command=self.parameter_canvas.yview)
        self.parameter_canvas.configure(yscrollcommand=parameter_scroll.set)
        parameter_scroll.pack(side='right',fill='y');self.parameter_canvas.pack(side='left',fill='both',expand=True)
        parameter=ttk.Frame(self.parameter_canvas,padding=12)
        parameter_window=self.parameter_canvas.create_window((0,0),window=parameter,anchor='nw')
        parameter.bind('<Configure>',lambda event:self.parameter_canvas.configure(scrollregion=self.parameter_canvas.bbox('all')))
        self.parameter_canvas.bind('<Configure>',lambda event:self.parameter_canvas.itemconfigure(parameter_window,width=event.width))
        self.results=ttk.Frame(self.notebook,padding=12)
        grains_page=ttk.Frame(self.notebook,padding=10)
        provenance=ttk.Frame(self.notebook,padding=10)
        for page,title in [(parameter_page,'参数设置'),(self.results,'力与图表'),(grains_page,'磨粒明细'),(provenance,'假设与来源')]:self.notebook.add(page,text=title)
        self.variables={};self.entries=[]
        for index,(name,label,_) in enumerate(FIELDS):
            col=(index%2)*2;row=index//2
            ttk.Label(parameter,text=label).grid(row=row,column=col,sticky='w',padx=(0,8),pady=5)
            v=tk.StringVar();self.variables[name]=v
            e=ttk.Entry(parameter,textvariable=v,width=21);e.grid(row=row,column=col+1,sticky='ew',padx=(0,22),pady=5)
            self.entries.append(e);v.trace_add('write',self._edited)
        parameter.columnconfigure(1,weight=1);parameter.columnconfigure(3,weight=1)
        for row,(name,label,choices) in enumerate([('boundary_rule','边界条件',BOUNDARIES),('formula_convention','公式约定',FORMULAS)],start=8):
            ttk.Label(parameter,text=label).grid(row=row,column=0,sticky='w',pady=5)
            v=tk.StringVar();self.variables[name]=v
            e=ttk.Combobox(parameter,textvariable=v,values=list(choices),state='readonly',width=28)
            e.grid(row=row,column=1,columnspan=3,sticky='ew',pady=5);self.entries.append(e);v.trace_add('write',self._edited)
        ttk.Label(parameter,text='材料：440C 文献参数。K₁ 参考值适用于记录的标定条件；修改工况时保持固定。\n输出为磨削力预测。真实接触、材料分离和有限元场计算保留在独立路线。',wraplength=950).grid(row=10,column=0,columnspan=4,sticky='w',pady=12)
        self.summary=tk.StringVar(value='尚未计算。请设置参数或加载算例。')
        ttk.Label(self.results,textvariable=self.summary,font=('Microsoft YaHei UI',12),wraplength=1050).pack(anchor='w')
        result_bar=ttk.Frame(self.results);result_bar.pack(fill='x',pady=8)
        self.folder_button=ttk.Button(result_bar,text='打开结果目录',command=self.open_folder,state='disabled');self.folder_button.pack(side='left')
        self.bridge_button=ttk.Button(result_bar,text='导出现有内核总力输入',command=self.bridge_dialog,state='disabled');self.bridge_button.pack(side='left',padx=8)
        self.image_label=ttk.Label(self.results);self.image_label.pack(fill='both',expand=True)
        ttk.Label(grains_page,text='显示前 500 颗磨粒；完整明细见结果目录 grains.csv（可直接导入 Excel / Origin）。').pack(anchor='w',pady=(0,7))
        columns=('id','depth','stage','normal','tangential')
        self.grains=ttk.Treeview(grains_page,columns=columns,show='headings',height=16)
        for key,title in zip(columns,['编号','有效切深 (μm)','阶段','法向力 (N)','切向力 (N)']):
            self.grains.heading(key,text=title);self.grains.column(key,width=145,anchor='center')
        sb=ttk.Scrollbar(grains_page,command=self.grains.yview);self.grains.configure(yscrollcommand=sb.set)
        sb.pack(side='right',fill='y');self.grains.pack(fill='both',expand=True)
        for name,label in [('wear_provenance','磨损系数来源（修改 K₁ 时须同步更新）'),('reconstruction_provenance','砂轮重建假设')]:
            ttk.Label(provenance,text=label).pack(anchor='w',pady=4)
            v=tk.StringVar();self.variables[name]=v
            entry=ttk.Entry(provenance,textvariable=v);entry.pack(fill='x',pady=4);self.entries.append(entry);v.trace_add('write',self._edited)
        self.details=tk.Text(provenance,height=16,wrap='word',font=('Microsoft YaHei UI',10))
        self.details.pack(fill='both',expand=True,pady=8);self.details.configure(state='disabled')
        self.status=tk.StringVar(value='就绪')
        self.progress=ttk.Progressbar(self.frame,mode='indeterminate');self.progress.pack(fill='x')
        ttk.Label(self.frame,textvariable=self.status,wraplength=1100).pack(anchor='w',pady=(7,0))
        self.set_case(default_case(calibrated=True))
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.after(80,self._poll)

    def values(self):
        d={k:v.get() for k,v in self.variables.items()}
        d['boundary_rule']=BOUNDARIES.get(d['boundary_rule'],d['boundary_rule'])
        d['formula_convention']=FORMULAS.get(d['formula_convention'],d['formula_convention'])
        return d

    def current_case(self):return form_to_case(self.values(),self.base_case)

    def set_case(self,case):
        values=case_to_form(case)
        self._setting=True;self.base_case=case
        for key,value in values.items():
            options=BOUNDARIES if key=='boundary_rule' else FORMULAS if key=='formula_convention' else {}
            self.variables[key].set(next((k for k,v in options.items() if v==value),value))
        self._setting=False;self.dirty=False
        self._refresh_state()

    def _edited(self,*args):
        if self._setting:return
        self.dirty=True;self._refresh_state()

    def _refresh_state(self):
        if self.result is None:self.result_state='none'
        else:
            try:fp=hashlib.sha256(canonical(self.current_case().to_dict()).encode('utf-8')).hexdigest()
            except (ValueError,OverflowError):fp=None
            self.result_state='current' if fp==self.result['input_fingerprint'] else 'stale'
        self.bridge_button.configure(state='normal' if self.result_state=='current' and not self.busy else 'disabled')
        if self.result is not None:
            r=self.result
            self.summary.set(f"Fn = {r['total_n_N']:.6g} N    Ft = {r['total_t_N']:.6g} N    有效磨粒 {r['counts']['dynamic']}\n"
                +('当前参数对应结果' if self.result_state=='current' else '历史结果：当前参数已改变，请重新计算')+' · 文献重建，尚无独立实验验证')

    def _error(self,error):
        self.last_error=str(error);self.status.set('操作未完成：'+str(error))

    def _busy(self,value):
        self.busy=value
        for control in self.controls+[self.run_button]:control.configure(state='disabled' if value else 'normal')
        for e in self.entries:e.configure(state='disabled' if value else 'readonly' if isinstance(e,ttk.Combobox) else 'normal')
        if value:self.started=time.monotonic();self.progress.start(12)
        else:self.progress.stop()
        self._refresh_state()

    def _submit(self,kind,operation):
        if self.busy:return
        self.last_error=None;self._busy(True)
        def work():
            try:self._events.put((kind,operation(),None))
            except Exception as exc:self._events.put((kind,None,str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def start_calculation(self):
        if self.busy:return
        try:case=self.current_case()
        except (ValueError,OverflowError) as exc:self._error(exc);return
        output=self.output_root/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        self.pending_output=output
        self._submit('result',lambda:(run_case(case,output),output,False))

    def _poll(self):
        try:
            kind,payload,error=self._events.get_nowait()
        except queue.Empty:
            if self.busy:self.status.set(f'正在计算 / 读取 / 发布结果… {time.monotonic()-self.started:.1f} s')
        else:
            self._busy(False)
            if error:self._error(error)
            else:
                try:
                    if kind=='result':self._display(*payload)
                    elif kind=='bridge':self.status.set('现有内核输入已导出：'+str(payload))
                except Exception as exc:self._error(exc)
        self.root.after(80,self._poll)

    def _display(self,result,directory,restore):
        from PIL import Image, ImageTk
        case=read_case(result['normalized_input'])
        if restore and isinstance(case,ReconstructionCase):self.set_case(case)
        self.result=result;self.result_directory=Path(directory)
        self.grains.delete(*self.grains.get_children())
        labels={'inactive':'未接触','ploughing':'耕犁','transition':'过渡','cutting':'切削'}
        for row in result['grains'][:500]:
            self.grains.insert('', 'end',values=(row['grain_id'],f"{row['depth_m']*1e6:.5g}",labels[row['stage']],f"{row['total_n_N']:.6g}",f"{row['total_t_N']:.6g}"))
        with Image.open(self.result_directory/'force_components.png') as image:
            image.thumbnail((max(650,self.root.winfo_width()-90),max(210,self.root.winfo_height()-340)))
            self._photo=ImageTk.PhotoImage(image,master=self.root)
        self.image_label.configure(image=self._photo)
        self.details.configure(state='normal');self.details.delete('1.0','end')
        self.details.insert('end','当前结果的完整假设和来源（随输入保存）：\n'+json.dumps(dict(
            source_doi=result['source_doi'],input=result['normalized_input'],limitations=result['limitations']),ensure_ascii=False,indent=2))
        self.details.configure(state='disabled')
        self.folder_button.configure(state='normal');self._refresh_state()
        self.status.set('结果已保存并严格回读：'+str(directory));self.notebook.select(self.results)

    def open_result_directory(self,directory):
        if self.busy:return
        path=Path(directory)
        self._submit('result',lambda:(read_result(path),path,True))

    def save_to_path(self,path):
        save_case(self.current_case(),path);self.dirty=False;self.status.set('算例已保存：'+str(path))

    def _discard(self):return not self.dirty or messagebox.askyesno('未保存参数','当前参数尚未另存，继续将替换编辑内容。',parent=self.root)

    def choose_default(self,calibrated):
        if not self.busy and self._discard():self.set_case(default_case(calibrated));self.status.set('算例已载入。')

    def load_dialog(self):
        if not self._discard():return
        path=filedialog.askopenfilename(parent=self.root,filetypes=[('JSON 算例','*.json')])
        if path:
            try:self.set_case(load_case(path));self.status.set('已加载：'+path)
            except (ValueError,OSError) as exc:self._error(exc)

    def save_dialog(self):
        path=filedialog.asksaveasfilename(parent=self.root,defaultextension='.json',filetypes=[('JSON 算例','*.json')],confirmoverwrite=False)
        if path:
            try:self.save_to_path(path)
            except (ValueError,OSError) as exc:self._error(exc)

    def open_dialog(self):
        if not self._discard():return
        path=filedialog.askdirectory(parent=self.root,title='选择含 summary.json 的结果目录')
        if path:self.open_result_directory(path)

    def open_folder(self):
        if self.result_directory:
            try:os.startfile(self.result_directory)
            except OSError as exc:self._error(exc)

    def bridge_dialog(self):
        if self.result_state!='current' or self.busy:return
        parent=filedialog.askdirectory(parent=self.root,title='选择导出位置（自动建立新子目录）')
        if not parent:return
        from .legacy_bridge import export_bridge
        output=Path(parent)/('literature_bridge_'+uuid.uuid4().hex[:8])
        source=self.result_directory
        def operation():export_bridge(source,output);return output
        self._submit('bridge',operation)

    def close(self):
        if self.busy:
            self.status.set('正在完成计算与结果保存，请等待后关闭。');return
        if self._discard():self.root.destroy()


def main():
    root=tk.Tk();Workbench(root);root.mainloop()


if __name__=='__main__':main()
