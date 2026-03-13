import React, { useEffect, useState } from 'react';
import { createProject, deleteProject, listProjects } from './services/api';

export default function ProjectSelector({ onOpenProject, onManageProjects }) {
  const [projects, setProjects] = useState([]);
  const [newProjectName, setNewProjectName] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState('');

  const loadProjects = async () => {
    setIsLoading(true);
    setError('');
    try {
      const result = await listProjects();
      setProjects(result.projects || []);
    } catch (e) {
      setError(e.message || 'Failed to load projects');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadProjects();
  }, []);

  const handleCreate = async () => {
    const name = newProjectName.trim();
    if (!name) return;
    setIsCreating(true);
    setError('');
    try {
      const result = await createProject(name);
      setNewProjectName('');
      await loadProjects();
      onOpenProject(result.projectId);
    } catch (e) {
      setError(e.message || 'Failed to create project');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDelete = async (projectId) => {
    if (!window.confirm('Delete this project and all artifacts?')) return;
    setError('');
    try {
      await deleteProject(projectId);
      await loadProjects();
    } catch (e) {
      setError(e.message || 'Failed to delete project');
    }
  };

  return (
    <div className="bg-white fixed inset-0 overflow-auto">
      <div className="h-[62px] bg-[#365D60] flex items-center justify-between px-[64px]">
        <div className="flex items-center gap-[12px]">
          <img src="/logo.png" alt="Logo" className="h-[40px] w-auto" />
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[24px] text-white">EcoMetrics</p>
        </div>
        <button
          onClick={onManageProjects}
          className="px-[18px] py-[8px] rounded-[10px] bg-white text-[#365D60] font-['Noto_Sans',sans-serif] text-[14px] font-semibold"
        >
          Project Management
        </button>
      </div>

      <div className="max-w-[980px] mx-auto px-[24px] py-[32px]">
        <h1 className="font-['Noto_Sans',sans-serif] font-bold text-[30px] text-[#13383B] mb-[8px]">Select Project</h1>
        <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E] mb-[24px]">
          Open an existing project or create a new one. Project names must be unique.
        </p>

        <div className="bg-[#F5F5F5] rounded-[10px] p-[18px] mb-[20px]">
          <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#13383B] mb-[10px]">Create New Project</p>
          <div className="flex gap-[10px]">
            <input
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="Project name"
              className="flex-1 h-[42px] rounded-[10px] border border-[#D9D9D9] px-[12px] font-['Noto_Sans',sans-serif] text-[14px]"
            />
            <button
              onClick={handleCreate}
              disabled={isCreating || !newProjectName.trim()}
              className="px-[18px] rounded-[10px] bg-[#29ABB5] text-white font-['Noto_Sans',sans-serif] text-[14px] disabled:opacity-60"
            >
              {isCreating ? 'Creating...' : 'Create'}
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-[16px] rounded-[10px] bg-[#FDECEC] text-[#A33A3A] px-[12px] py-[10px] font-['Noto_Sans',sans-serif] text-[13px]">
            {error}
          </div>
        )}

        <div className="bg-white border border-[#EEEEEE] rounded-[10px] overflow-hidden">
          <div className="px-[16px] py-[12px] bg-[#D9D9D9] font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#13383B]">
            Existing Projects
          </div>
          {isLoading ? (
            <div className="p-[18px] font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">Loading...</div>
          ) : projects.length === 0 ? (
            <div className="p-[18px] font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">No projects yet.</div>
          ) : (
            <div>
              {projects.map((project) => (
                <div key={project.projectId} className="px-[16px] py-[12px] border-t border-[#EEEEEE] flex items-center justify-between gap-[12px]">
                  <div className="min-w-0">
                    <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#13383B] truncate">{project.name}</p>
                    <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] truncate">Updated: {project.lastModified || '-'}</p>
                  </div>
                  <div className="flex items-center gap-[8px]">
                    <button
                      onClick={() => onOpenProject(project.projectId)}
                      className="px-[12px] h-[34px] rounded-[8px] bg-[#29ABB5] text-white font-['Noto_Sans',sans-serif] text-[13px]"
                    >
                      Open
                    </button>
                    <button
                      onClick={() => handleDelete(project.projectId)}
                      className="px-[12px] h-[34px] rounded-[8px] bg-[#FDECEC] text-[#A33A3A] font-['Noto_Sans',sans-serif] text-[13px]"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
