
import React, { useState } from 'react';

type MenuItem = {
  label: string;
  items?: string[];
  isActive?: boolean;
};

const menuItems: MenuItem[] = [
  { label: 'File', items: ['New Notebook', 'Open', 'Save', 'Save As', 'Close and Halt'] },
  { label: 'Edit', items: ['Cut Cells', 'Copy Cells', 'Paste Cells', 'Delete Cells', 'Split Cell'] },
  { label: 'View', items: ['Toggle Header', 'Toggle Toolbar', 'Toggle Line Numbers'] },
  { label: 'Insert', items: ['Insert Cell Above', 'Insert Cell Below'] },
  { label: 'Cell', items: ['Run', 'Run All', 'Run All Above', 'Run All Below'] },
  { label: 'Kernel', items: ['Interrupt', 'Restart', 'Restart & Clear Output', 'Restart & Run All'] },
  { label: 'Widgets', items: ['Save Notebook Widget State', 'Clear Notebook Widget State'] },
  { label: 'Help', items: ['User Interface Tour', 'Keyboard Shortcuts', 'About'] },
];

const JupyterMenu: React.FC = () => {
  const [activeMenu, setActiveMenu] = useState<string | null>(null);

  const handleMenuClick = (label: string) => {
    if (activeMenu === label) {
      setActiveMenu(null);
    } else {
      setActiveMenu(label);
    }
  };

  const handleMenuItemClick = (item: string) => {
    console.log(`Menu item clicked: ${item}`);
    setActiveMenu(null);
  };

  return (
    <div className="flex items-center bg-jupyter-menubar border-b border-jupyter-border h-10">
      {menuItems.map((menu) => (
        <div key={menu.label} className="relative">
          <button
            className={`px-4 h-full hover:bg-gray-200 ${activeMenu === menu.label ? 'bg-gray-200' : ''}`}
            onClick={() => handleMenuClick(menu.label)}
          >
            {menu.label}
          </button>
          {activeMenu === menu.label && menu.items && (
            <div className="jupyter-dropdown absolute left-0 min-w-[200px] animate-fade-in">
              {menu.items.map((item) => (
                <div
                  key={item}
                  className="jupyter-dropdown-item"
                  onClick={() => handleMenuItemClick(item)}
                >
                  {item}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
      <div className="ml-auto flex items-center">
        <div className="border border-gray-300 rounded-sm px-2 py-1 mr-2 text-sm bg-white">
          Trusted
        </div>
        <div className="border border-gray-300 rounded-sm px-2 py-1 mr-2 text-sm bg-white">
          Python 3 (ipykernel)
        </div>
        <div className="w-5 h-5 rounded-full bg-gray-300 flex items-center justify-center mr-2">
          ?
        </div>
      </div>
    </div>
  );
};

export default JupyterMenu;
