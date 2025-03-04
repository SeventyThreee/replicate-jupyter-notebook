
import React, { useState } from 'react';
import JupyterHeader from '@/components/JupyterHeader';
import JupyterMenu from '@/components/JupyterMenu';
import JupyterToolbar from '@/components/JupyterToolbar';
import CodeCell from '@/components/CodeCell';

interface CellData {
  id: number;
  code: string;
  cellType: 'code' | 'markdown';
}

const Index = () => {
  // Initial cell data
  const initialCells: CellData[] = [
    { id: 1, code: `<h1>Interactive Python Visualization Libraries</h1>`, cellType: 'markdown' },
    { id: 2, code: `import numpy as np
import plotly.offline as py
import plotly.figure_factory as ff
from bokeh.models import HoverTool, WheelZoomTool
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
output_notebook()`, cellType: 'code' },
    { id: 3, code: `<h2>Plotly</h2>`, cellType: 'markdown' },
    { id: 4, code: `py.init_notebook_mode()
t = np.linspace(-1, 1.2, 2000)
x = (t**3) + (0.3 * np.random.randn(2000))
y = (t**6) + (0.3 * np.random.randn(2000))

colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]

fig = ff.create_2d_density(
    x, y, colorscale=colorscale,
    hist_color='rgb(255, 237, 222)', point_size=3
)

py.iplot(fig, filename='histogram_subplots')`, cellType: 'code' },
    { id: 5, code: `<h2>Bokeh</h2>`, cellType: 'markdown' },
    { id: 6, code: `n = 500
x = 2 + 2*np.random.standard_normal(n)
y = 2 + 2*np.random.standard_normal(n)

p = figure(title="Hexbin for 500 points", match_aspect=True,
         tools="wheel_zoom,pan,reset", background_fill_color='#440154')
p.grid.visible = False

r, bins = p.hexbin(x, y, size=0.5, hover_color="pink", hover_alpha=0.8)`, cellType: 'code' }
  ];

  // State for cells and active cell
  const [cells, setCells] = useState<CellData[]>(initialCells);
  const [activeCellIndex, setActiveCellIndex] = useState<number | null>(null);

  // Handle cell focus
  const handleCellFocus = (index: number) => {
    setActiveCellIndex(index);
  };

  // Handle Alt+Enter to create a new cell
  const handleAltEnter = (index: number) => {
    const newCellId = Math.max(...cells.map(cell => cell.id)) + 1;
    const newCell: CellData = {
      id: newCellId,
      code: '',
      cellType: 'code'
    };
    
    // Insert the new cell after the current one
    const updatedCells = [
      ...cells.slice(0, index + 1),
      newCell,
      ...cells.slice(index + 1)
    ];
    
    setCells(updatedCells);
    // Set focus to the new cell
    setActiveCellIndex(index + 1);
  };

  return (
    <div className="min-h-screen flex flex-col bg-white">
      <JupyterHeader />
      <JupyterMenu />
      <JupyterToolbar />
      
      <div className="flex-1 px-4 py-2 overflow-y-auto">
        {cells.map((cell, index) => (
          <CodeCell 
            key={cell.id}
            cellNumber={cell.cellType === 'code' ? cell.id : 0}
            code={cell.code} 
            cellType={cell.cellType} 
            isActive={activeCellIndex === index}
            onFocus={() => handleCellFocus(index)}
            onAltEnter={() => handleAltEnter(index)}
          />
        ))}
      </div>
    </div>
  );
};

export default Index;
