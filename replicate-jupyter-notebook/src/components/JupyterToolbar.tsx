
import React from 'react';
import { File, Save, Plus, Scissors, Copy, ChevronUp, ChevronDown, Play, Square, ArrowLeft, ArrowRight } from 'lucide-react';

const JupyterToolbar: React.FC = () => {
  return (
    <div className="flex items-center p-1 bg-jupyter-toolbar border-b border-jupyter-border">
      <div className="flex items-center border-r border-jupyter-border pr-1">
        <button className="jupyter-toolbar-button" title="Save and Checkpoint">
          <Save size={16} />
        </button>
        <button className="jupyter-toolbar-button" title="Insert Cell Below">
          <Plus size={16} />
        </button>
        <button className="jupyter-toolbar-button" title="Cut Selected Cells">
          <Scissors size={16} />
        </button>
        <button className="jupyter-toolbar-button" title="Copy Selected Cells">
          <Copy size={16} />
        </button>
        <button className="jupyter-toolbar-button" title="Paste Cells Below">
          <File size={16} />
        </button>
      </div>
      
      <div className="flex items-center border-r border-jupyter-border px-1">
        <button className="jupyter-toolbar-button" title="Move Selected Cells Up">
          <ChevronUp size={16} />
        </button>
        <button className="jupyter-toolbar-button" title="Move Selected Cells Down">
          <ChevronDown size={16} />
        </button>
      </div>
      
      <div className="flex items-center border-r border-jupyter-border px-1">
        <button className="jupyter-toolbar-button flex items-center justify-center gap-1 px-2 w-auto" title="Run">
          <Play size={14} />
          <span className="text-sm">Run</span>
        </button>
        <button className="jupyter-toolbar-button" title="Interrupt Kernel">
          <Square size={14} />
        </button>
        <button className="jupyter-toolbar-button" title="Restart Kernel">
          <ArrowRight size={14} />
        </button>
        <button className="jupyter-toolbar-button" title="Restart Kernel and Run All Cells">
          <ArrowLeft size={14} />
        </button>
      </div>
      
      <div className="flex items-center ml-2">
        <select className="h-8 px-2 border border-jupyter-border rounded-sm bg-white text-sm">
          <option>Markdown</option>
          <option>Code</option>
          <option>Raw</option>
        </select>
      </div>
    </div>
  );
};

export default JupyterToolbar;
