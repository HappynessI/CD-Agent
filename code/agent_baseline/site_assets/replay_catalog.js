(() => {
  document.documentElement.classList.add('has-js');
  const form = document.getElementById('catalog-filters');
  const rows = [...document.querySelectorAll('[data-replay-row]')];
  const names = ['task', 'constraint', 'dataset', 'harness'];
  const status = document.getElementById('filter-status');
  const empty = document.getElementById('filter-empty');
  const url = new URL(location.href);

  for (const name of names) {
    const control = form.elements.namedItem(name);
    const value = url.searchParams.get(name);
    if ([...control.options].some(option => option.value === value)) control.value = value;
  }

  function update() {
    let visible = 0;
    for (const row of rows) {
      row.hidden = names.some(name => {
        const value = form.elements.namedItem(name).value;
        return value && row.dataset[name] !== value;
      });
      if (!row.hidden) visible += 1;
    }
    status.textContent = `显示 ${visible} / ${rows.length} 个会话`;
    empty.hidden = visible > 0;
  }

  form.addEventListener('change', update);
  form.addEventListener('submit', event => event.preventDefault());
  form.addEventListener('reset', () => setTimeout(update));
  for (const link of document.querySelectorAll('[data-coverage-task]')) {
    link.addEventListener('click', () => {
      for (const name of names) form.elements.namedItem(name).value = '';
      form.elements.namedItem('task').value = link.dataset.coverageTask;
      form.elements.namedItem('constraint').value = link.dataset.coverageConstraint;
      update();
    });
  }
  update();
})();
