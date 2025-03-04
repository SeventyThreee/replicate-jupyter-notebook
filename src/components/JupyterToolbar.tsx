
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
        
        <button className="jupyter-toolbar-button ml-2" title="Command Mode">
          <div className="w-5 h-5 bg-gray-300 rounded-sm"></div>
        </button>
        
        <button className="jupyter-toolbar-button" title="Edit Mode">
          <div className="w-5 h-5 flex items-center justify-center">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 4L19 20L12 16L5 20L12 4Z" stroke="#4285F4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        </button>
      </div>
    </div>
  );
};

export default JupyterToolbar;
