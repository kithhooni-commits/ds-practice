// humanoid.js / effects.js 가 실제로 쓰는 THREE API만 흉내낸 최소 스텁 (헤드리스 검증용)
class V3 {
  constructor(x=0,y=0,z=0){this.x=x;this.y=y;this.z=z;}
  set(x,y,z){this.x=x;this.y=y;this.z=z;return this;}
  setScalar(v){this.x=v;this.y=v;this.z=v;return this;}
}
class Euler { constructor(){this.x=0;this.y=0;this.z=0;} }
class Obj3D {
  constructor(){ this.position=new V3(); this.rotation=new Euler(); this.scale=new V3(1,1,1); this.children=[]; this.visible=true; }
  add(c){ this.children.push(c); }
}
class Group extends Obj3D {}
class Mesh extends Obj3D { constructor(g,m){ super(); this.geometry=g; this.material=m; } }
function geo(){ return {}; }
// THREE.Color 흉내 — humanoid 가 setHex 로 두개골 색을 바꾼다
class Color {
  constructor(hex){ this.hex = hex || 0; }
  setHex(h){ this.hex = h; return this; }
  getHex(){ return this.hex; }
}
function mat(o){
  const m = Object.assign(this||{}, { opacity:1, transparent:false }, o||{});
  // color/emissive 는 숫자로 넘어오지만 실제 THREE 에서는 Color 객체다
  m.color = new Color(typeof (o&&o.color) === 'number' ? o.color : 0xffffff);
  m.emissive = new Color(typeof (o&&o.emissive) === 'number' ? o.emissive : 0x000000);
  m.dispose = m.dispose || function(){};
  return m;
}
global.THREE = {
  Group, Mesh, Vector3: V3,
  CylinderGeometry: geo, SphereGeometry: geo, BoxGeometry: geo,
  CanvasTexture: function(){ return this; },
  SpriteMaterial: function(o){ return Object.assign(this, { opacity:1, rotation:0,
                                dispose(){}, color: new Color((o&&o.color)||0xffffff) }, o||{},
                                { color: new Color((o&&o.color)||0xffffff) }); },
  Sprite: class extends Mesh { constructor(m){ super(null, m); this.isSprite = true; } },
  AdditiveBlending: 2, NormalBlending: 1,
  LatheGeometry: geo, TorusGeometry: geo, PlaneGeometry: geo,
  Vector2: function(x, y){ this.x = x; this.y = y; return this; },
  MeshStandardMaterial: mat, MeshBasicMaterial: mat,
};
global.window = global;

// 캔버스 스텁 — humanoid 가 불꽃 오라 텍스처를 캔버스에 그린다
const _ctx2d = new Proxy({}, {
  get(_, k) {
    if (k === 'createLinearGradient' || k === 'createRadialGradient')
      return () => ({ addColorStop() {} });
    if (k === 'measureText') return (t) => ({ width: String(t).length * 10 });
    return () => {};
  },
  set() { return true; },
});
if (!global.document) {
  global.document = {
    createElement: () => ({ width: 0, height: 0, getContext: () => _ctx2d,
                            toDataURL: () => 'data:image/png;base64,AAAA' }),
  };
}
global.performance = { now: () => Date.now() };
