const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const site=path.resolve(process.argv[2]); const browser=await chromium.launch({headless:true});
 try {
  const page=await browser.newPage({offline:true,viewport:{width:1500,height:1000}});
  const errors=[],remote=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))remote.push(r.url())});
  await page.goto(pathToFileURL(path.join(site,'index.html')).href);
  await page.locator('img[src]').evaluateAll(async imgs=>{for(const im of imgs)await im.decode()});
  assert.equal(await page.locator('img[src]').evaluateAll(imgs=>imgs.filter(i=>!i.complete||!i.naturalWidth).length),0);
  await page.screenshot({path:path.join(site,'validation/desktop.png')});
  await page.locator('#next').click();assert((await page.locator('#position').textContent()).includes('Step'));
  await page.locator('.overview .zoom').first().click();assert(await page.locator('dialog').isVisible());await page.keyboard.press('Escape');
  const funnel=page.getByRole('heading',{name:'find_missed 漏斗机制可视化拆解'}).first();await funnel.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(site,'validation/funnel.png')});
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.scrollTo(0,0));
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'Mobile overflow');
  await page.screenshot({path:path.join(site,'validation/mobile.png')});
  assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
  fs.writeFileSync(path.join(site,'validation/browser.json'),JSON.stringify({status:'passed',offline:true,images:await page.locator('img[src]').count(),remoteRequests:remote,errors,checks:['all image decode','local file URL','timeline navigation','zoom','mobile width']},null,2));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
