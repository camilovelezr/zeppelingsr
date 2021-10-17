import pandas as pd
import numpy as np
import glob
from periodictable import elements
from dateutil.parser import parse
import json
import pytz
import getpass
import cordra
from uuid import uuid4
from alive_progress import alive_bar
import itertools
import re


# dictionary to rename dataframes' columns
rename_columns = {"PART#": "NUMBER",
"PARTNUM": "NUMBER",
"FIELD#": "FIELD",
"FIELDNUM": "FIELD",
"MAGFIELD#": "MAGFIELD",
"MAGFIELDNUM": "MAGFIELD",
"X_ABS": "XABS",
"Y_ABS": "YABS",
"X_DAC": "XDAC",
"Y_DAC": "YDAC",
"XCENT": "XDAC",
"YCENT": "YDAC",
"X_FERET": "XFERET",
"Y_FERET": "YFERET",
"DVG": "DAVG",
"DAVE": "DAVG",
"PERIM": "PERIMETER",
"ORIENT": "ORIENTATION",
"LIVE_TIME": "LIVETIME",
"FIT_QUAL": "FITQUAL",
"MAG_INDEX": "MAGINDEX",
"FIRST_ELEM": "FIRSTELM",
"SECOND_ELEM": "SECONDELM",
"THIRD_ELEM": "THIRDELM",
"FOURTH_ELEM": "FOURTHELM",
"ATOMICNUMBER1": "FIRSTELM",
"ATOMICNUMBER2": "SECONDELM",
"ATOMICNUMBER3": "THIRDELM",
"ATOMICNUMBER4": "FOURTHELM",
"FIRST_CONC": "COUNTS1",
"SECOND_CONC": "COUNTS2",
"THIRD_CONC": "COUNTS3",
"FOURTH_CONC": "COUNTS4",
"FIRST_PCT": "FIRSTPCT",
"SECOND_PCT": "SECONDPCT",
"THIRD_PCT": "THIRDPCT",
"FOURTH_PCT": "FOURTHPCT",
"PCT1": "FIRSTPCT",
"PCT2": "SECONDPCT",
"PCT3": "THIRDPCT",
"PCT4": "FOURTHPCT",
"TYPE(4ET)#": "TYPE4ET",
"TYPE(4ET)": "TYPE4ET",
"VOID_AREA": "VOIDAREA",
"RMS_VIDEO": "RMSVIDEO",
"VERIFIED_CLASS": "VERIFIEDCLASS",
"EDGE_ROUGHNESS": "EDGEROUGHNESS",
"COMP_HASH": "COMPHASH",
"PSEM_CLASS": "CLASS"}


"""
Functions
"""

def CleanName(name): # take hdz path and return clean version
    return name.split('/')[-1].replace(" ", "_").replace("[", "_").replace("]",
                                                                           "_").replace("+",
                                                                                        "_plus_").replace('data', 'raw').split('.')[0]



class ZeppelinDataset:
    # RE used to search for keywords
    _pp = re.compile('^PARTICLE_PARAMETERS=')
    _cl = re.compile('^CLASS\d{1,2}=')
    _colem = re.compile('_ELEM$')
    # morphology parameters
    _morph = ["NUMBER", "XABS", "YABS", "DAVG", "DAVE", "DMIN", "DMAX", "DPERP", "PERIMETER", "ORIENTATION", "AREA"]

    def __init__(self, hdz):
        self.hdz = hdz
        self.pxz = f"{hdz.replace('.hdz', '.pxz')}"
        self.df = pd.read_csv(self.pxz, sep = '\t', header=None, low_memory=False)
        self.hdzclean = CleanName(self.hdz) # no path, no extension
        with open(hdz, 'r', encoding = 'cp1252') as file:
            h = np.array(file.read().replace('\t \t', '\t 1\t').split())
            first = (np.where(np.array([bool(re.search(self._pp, x)) for x in h]))[0][0]+1)
            c = itertools.takewhile(lambda x: x < len(h), itertools.count(first, 3))
            columns = h[list(c)]
            file.seek(0)
            self._meta = np.array(file.read().split('\n'))
            t = list(itertools.takewhile(lambda x: not bool(re.search(self._pp, x)), self._meta)) # take until re match pp
            t.append(self._meta[len(t)])

        self.df.columns = columns
        classes = dict((map(lambda x: x.replace('CLASS', '').split('=', 1),
          (itertools.filterfalse(lambda x: not bool(self._cl.match(x)), t)))))
        self.df['PSEM_CLASS'] = (self.df['PSEM_CLASS'].apply(lambda x: str(x))).map(classes)
        self.header = dict(map(lambda x: x.split('=', 1), t)) #metadata
        # dictionary of elements. <atomic number>:<symbol>
        element = {el.number: el.symbol for el in elements}
        element_upper = {el.number: (el.symbol).upper() for el in elements}
        el_cols = itertools.filterfalse(lambda x: not bool(self._colem.search(x)), columns)
        for x in el_cols:
            self.df[x] = self.df[x].map(element)

        # Normalizing elements symbols
        mapper = {element_upper[n] : element[n] for n in range(119)}
        mapper2 = {'U['+element_upper[n]+']' : 'U['+element[n]+']' for n in range(119)}

        # renaming columns dictionary
        self.df.rename(columns = mapper, inplace = True)
        self.df.rename(columns = mapper2, inplace = True)
        self.df.rename(columns = rename_columns, inplace = True)

        self.header['ANALYSIS_DATE'] = parse(self.header['ANALYSIS_DATE']).strftime('%Y-%m-%d')
        self.header['ACCELERATING_VOLTAGE'] = {'value': float(self.header['ACCELERATING_VOLTAGE'].split()[0]),
                                               'unitText':self.header['ACCELERATING_VOLTAGE'].split()[1]}
        self.header['PROBE_CURRENT'] = {'value': float(self.header['PROBE_CURRENT'].split()[0]),
                                        'unitText':self.header['PROBE_CURRENT'].split()[1]}
        self.header['WORKING_DISTANCE'] = {'value': float(self.header['WORKING_DISTANCE']),
                                           'unitText': 'mm'}
        x = parse(self.header['ANALYSIS_DATE'] + ' ' +self.header['START_TIME'], tzinfos = {'EDT': pytz.timezone('EST5EDT'), "EST":pytz.timezone("EST"),
                                                                                            "MDT":pytz.timezone("MST7MDT"), "MST":pytz.timezone("MST")})
        self.header['DATETIME'] = x.astimezone(pytz.utc).isoformat()
        # Mapping MAG0 and MAG_FMT as key value pairs
        self.header[self.header['MAG_FMT'].split()[0].upper()+'_DATA'] = {'value': float(self.header['MAG0'].split()[0]), 'unitText': 'Assuming a 3.5 in field of view'}

        self.header[self.header['MAG_FMT'].split()[1].upper()] = {'value': float(self.header['MAG0'].split()[1]), 'unitText': ''}

        self.header[self.header['MAG_FMT'].split()[2].upper()] = {'value': float(self.header['MAG0'].split()[2]), 'unitText': ''}

        self.header[self.header['MAG_FMT'].split()[3].upper()] = {'value': float(self.header['MAG0'].split()[3]), 'unitText': 'min'}

        self.header[self.header['MAG_FMT'].split()[4].upper()+'_DATA'] = {'value': float(self.header['MAG0'].split()[4]), 'unitText': 'sq mm'}

        self.df.sort_values(['NUMBER'], inplace=True)

    def json(self, row): # create JSON of particle
        r = (self.df.iloc[row-1]).to_json(orient = 'columns')
        r = json.loads(r)
        return r

    def json_metadata(self, row): # create JSON of particle with metadata attached
        r = (self.df.iloc[row-1]).to_json(orient = 'columns')
        r = json.loads(r)
        f = self.header.copy()
        f.update(r)
        return f
    def morphology(self):
        l = [t for t in self._morph if t in self.df.columns]
        return self.df[l]

    def _uploadheader(self, Zeppelin, CordraSession): # uploads metadata
        r = self.header
        name = self.hdzclean
        r.update({"CamiloExplore": 1, "name": f"{name.capitalize()} {Zeppelin.sample}",
                  "material": [Zeppelin.CordraId.sample]})
        response = cordra.CordraObject.create(
        CordraSession.host,
        r,
        obj_type = "Dataset",
        suffix = Zeppelin.hdznames[name],
        token = CordraSession.token,
        verify=False,
        acls=CordraSession.acls,
        )
    def _uploadparticles(self, Zeppelin, CordraSession, maxim = None): # uploads first n particles (if no n, all)
        if maxim==None:
            num = len(Zeppelin._datahdz.df)
        else:
            num = maxim
            df_up = self.df[self.df["NUMBER"]<=num]    
        for x in range(1, num+1):
            try:
                r = self.json(x)
                n = int(r['NUMBER'])
                s = f"{Zeppelin.suffix}/{self.hdzclean}/{n}"
                r.update({"isPartOf":[f"{Zeppelin.CordraId.sample}/{self.hdzclean}"], "CamiloExplore": 1,
                         "isBasedOn":[f"{Zeppelin.CordraId.sample}/image/{n}"], "material":[f"{Zeppelin.CordraId.sample}/{n}"]})
                response = cordra.CordraObject.create(
                CordraSession.host,
                r,
                obj_type = "Dataset",
                suffix = s,
                token = CordraSession.token,
                verify=False,
                acls=CordraSession.acls
                )
                Zeppelin.CordraId.particles[self.hdzclean].append(f"{CordraSession.prefix}/{s}")
            except BaseException as b:
                Zeppelin.problemparticles[f"Particle {n} of {self.hdzclean}"] = (b.response.text).split(":")[1].replace("}", "")
        Zeppelin.CordraId.particles[self.hdzclean] = set(Zeppelin.CordraId.particles[self.hdzclean])


class Zeppelin:
    class CordraId: # sub-class to save Cordra Ids of uploaded objects
        particles = dict() # initialize particle ids as a dictionary, keys will be reprocessings names
        pass

    def __init__(self, path):
        self.path = glob.glob(f"{path}/**/*data.hdz", recursive = True)[0].split('/data.hdz')[0]
        self.hdzs = sorted((glob.glob(f"{self.path}/*.hdz")))
        self.datasets = []
        # initialize attributes to save possible errors and exceptions
        self.missingpxz = []
        self.problemparticles = dict()

        for x in self.hdzs:
            try:
                self.datasets.append(ZeppelinDataset(x)) # create ZeppelinDataset objects for each hdz in Zeppelin
            except FileNotFoundError:
                self.missingpxz.append(x) # if no PXZ found, appends to attribute missingpxz
                print(f'Could not find {x.replace(".hdz", ".pxz")}.'+
                    ' To see which hdz files are missing their respective pxz files, see missingpxz attribute.')
        self.hdzs = list(itertools.filterfalse(lambda x: x in self.missingpxz, self.hdzs)) # remove missing pxzs from hdz list
        self.images = sorted(glob.glob(f"{self.path}/MAG0/*.tif")) # load image file names
        def getuuid():
            return str(uuid4())[14:18]
        self.uuid = getuuid() # get 4 characters from uuid4 string
        #while uuid in uuid_dict.values():
            #uuid = getuuid()
        def getdatahdz():
            data = np.where((map(lambda x: bool(re.search('data.hdz$', x)), self.hdzs)))[0][0]
            setattr(self, '_datahdz', self.datasets[data]) # sets _datahdz attribute as a reference to ZeppelinDataset of data.hdz (same memory position)
        def getyear(): # gets year from data.hdz
            getdatahdz()
            year = parse(self._datahdz.header['ANALYSIS_DATE']).year
            setattr(self, '_year', year)
        getyear()
        def hdznames(): # create dictionary used to generate CordraIds for metadata and particle entries
            t = list(map(lambda x: CleanName(x), self.hdzs))
            k = list(map(lambda x: f"gsr-{self._year}-{self.uuid}/{x}", t))
            d = dict(zip(t,k))
            setattr(self,'hdznames', d)
        hdznames()
        self.sample = self._datahdz.header['SAMPLE'] # Name of the sample
        self.nparticles = len(self._datahdz.df) # Number of particles in sample
        self.suffix = f"gsr-{self._year}-{self.uuid}" # suffix used for Cordra ids

        """""
        Initialize CordraId.particles to arrays with reprocessing names as keys
        """""
        for x in self.hdznames.keys():
            self.CordraId.particles[x] = []


    def upload_sample(self, CordraSession):
        r = {'Sample Type':'GSR Sample', "Particles": self.nparticles, "CamiloExplore": 1,
            "name": f"GSR Sample {self.sample}"}
        """
            if local != 'y': # add Nicholas as accountablePerson in sandbox
            r.update({'accountablePerson': [CordraSession._nicholas]})

            need to add this possibility
        """
        response = cordra.CordraObject.create(
        CordraSession.host,
        r,
        obj_type = "Material",
        suffix = self.suffix,
        token = CordraSession.token,
        verify=False,
        acls=CordraSession.acls,
        )
        _id = f"{CordraSession.prefix}/{self.suffix}"
        print(f"Succesfully uploaded sample with id {_id}")
        self.CordraId.sample = _id


    def upload_raw_particles(self, CordraSession, maxim=None):
        def sample_haspart(self, CordraSession):
            response = cordra.CordraObject.update(
                CordraSession.host,
                obj_id = self.CordraId.sample,
                obj_json=self.CordraId.rawparticles,
                jsonPointer="/hasPart",
                token = CordraSession.token,
                verify = False,
                full = True
            )
        self.CordraId.rawparticles = []
        if maxim==None:
            num = self.nparticles
        else:
            num = maxim
        particles = (map(lambda x: dict(x[1]) ,self._datahdz.morphology().iterrows()))
        with alive_bar(num, title="Uploading particles and updating sample object", bar='smooth') as bar:
            for x in itertools.islice(particles, num):
                try:
                    n = int(x['NUMBER'])
                    s = f"{self.suffix}/{n}"
                    x.update({"isPartOf":[self.CordraId.sample], "CamiloExplore":1, "name": f"Particle #{n} {self.sample}"})
                    response = cordra.CordraObject.create(
                    CordraSession.host,
                    x,
                    obj_type = "Material",
                    suffix = s,
                    token = CordraSession.token,
                    verify=False,
                    acls=CordraSession.acls,
                    )
                    self.CordraId.rawparticles.append(f"{CordraSession.prefix}/{s}")
                    bar()
                except:
                    print(f"There was an error with particle # {n}")
            sample_haspart(self, CordraSession)
        print(f"Finished uploading the first {num} particles. \n")

    def upload_images(self, CordraSession, maxim=None):
        self.CordraId.images = []
        if maxim==None:
            num = self.nparticles
        else:
            num = maxim
        with alive_bar(maxim, title="Uploading particle images", bar='smooth') as bar:
            for n in range(num):
                try:
                    file = self.images[n]
                    with open(file, 'rb') as image:
                        s = f"{self.suffix}/image/{int(n+1)}"
                        response = cordra.CordraObject.create(
                        CordraSession.host,
                        {"name": f"Image Particle #{n+1} {self.sample}", 'ParticleNumber':n+1, "ParticleType": "GSR", "CamiloExplore": 1,
                            "material": [f"{self.CordraId.sample}/{n+1}"]},
                        obj_type = "File",
                        payloads = {'Image': (file, image)},
                        suffix = s,
                        token = CordraSession.token,
                        verify=False,
                        acls=CordraSession.acls,
                        )
                    bar()
                    self.CordraId.images.append(f"{CordraSession.prefix}/{s}")
                except:
                    print(f"There was an error with particle # {n}")
        print(f"Finished uploading the first {maxim} particle images. \n")

    def upload_metadata(self, CordraSession):
        with alive_bar(len(self.datasets), title = "Uploading reprocessings metadata", bar='smooth') as bar:
            for x in self.datasets:
                x._uploadheader(self, CordraSession)
                bar()
        self.CordraId.metadata = dict()
        for x,y in self.hdznames.items():
            self.CordraId.metadata[x] = f"{CordraSession.prefix}/{y}"
        print(f"Succesfully uploaded the reprocessings metadata. \n")
    def upload_particles(self, CordraSession, maxim = None):
        def metadata_haspart(self, CordraSession, ZeppelinDataset):
            print(self.CordraId.particles[ZeppelinDataset.hdzclean])
            try:
                response = cordra.CordraObject.update(
                    CordraSession.host,
                    obj_id = f"{self.CordraId.metadata[ZeppelinDataset.hdzclean]}",
                    obj_json=self.CordraId.particles[ZeppelinDataset.hdzclean],
                    jsonPointer="/hasPart",
                    token = CordraSession.token,
                    verify = False,
                    full = True
                )
            except BaseException as e:
                print(f"Problem with {self.CordraId.metadata[ZeppelinDataset.hdzclean]}. Problem is {e}")
        if maxim==None:
            count = self.nparticles
        else:
            count = maxim
        with alive_bar(len(self.datasets), title = "Uploading particles", bar='smooth') as bar:
            for x in self.datasets:
                x._uploadparticles(self, CordraSession, count)
                metadata_haspart(self, CordraSession, x)
                bar()
        print(f"Finished uploading the first {count} particles. \n")
    def upload_all(self, CordraSession, maxim = None):
        if maxim==None:
            maxim = self.nparticles
        self.upload_sample(CordraSession)
        self.upload_raw_particles(CordraSession, maxim)
        self.upload_images(CordraSession, maxim)
        self.upload_metadata(CordraSession)
        self.upload_particles(CordraSession, maxim)

class CordraSession:
    _nicholas = "prefix/af33681656fc73e15b0e"
    _zach = "prefix/fa397449de02f4c77fbb"
    _camilo = "prefix/e2cc6ac92e2e50957fd9"
    """
        if local !='y':
        acl["writers"] = [_nicholas, _zach, _camilo]
    """
    def __init__(self, host=None, username=None, password=None, prefix=None, acls=None):
        if host == None:
            host = input("Insert your cordra host link: ")
        if username == None:
            username = input("Insert your cordra username: ")
        if password == None:
            password = getpass.getpass("Password: ")
        if prefix == None:
            prefix = input("Enter cordra's prefix: ")
        def _getcordrainfo(self):
            token = cordra.Token.create(host,username,password,verify=False)
            setattr(self, 'token', token)
            setattr(self, 'prefix', prefix)
            setattr(self, 'username', username)
            setattr(self, 'host', host)
            setattr(self, 'acls', acls)
        _getcordrainfo(self)
    def restore_token(self):
        password = getpass.getpass("Password: ")
        token = cordra.Token.create(self.host,self.username,password,verify=False)
        setattr(self, 'token', token)
