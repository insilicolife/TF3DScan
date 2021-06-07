import numpy as np
import pandas as pa
import requests, sys
import json 
from Bio.Seq import Seq
import os


class TF3DScan:
    def __init__(self,genes,PWM_directory,seqs=None):
        self.gene_names=genes
        self.PWM_dir=PWM_directory
        self.seq=None
        self.PWM=None
        self.weights=None
        self.proteins=None
        self.initialize()
        
    def initialize(self):
        self.seq=self.get_seq_by_name(self.gene_names)
        self.PWM=self.convolutional_filter_for_each_TF(self.PWM_dir)
        self.weights, self.proteins= self.get_Weights(self.PWM)
        return 
    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return (e_x / e_x.sum(axis=0))
    def convolutional_filter_for_each_TF(self,PWM_directory):
        path = PWM_directory
        #print(path)
        filelist = os.listdir(path)
        TF_kernel_PWM={}
        for file in filelist:
            TF_kernel_PWM[file.split("_")[0]] = pa.read_csv(path+file, sep="\t", skiprows=[0], header=None)
        return TF_kernel_PWM
    
    def get_reverse_scaning_weights(self, weight):
        return np.flipud(weight[:,[3,2,1,0]])
    
    def get_Weights(self, filter_PWM_human):
        #forward and reverse scanning matrix with reverse complement
        #forward_and_reverse_direction_filter_list=[{k:np.dstack((filter_PWM_human[k],self.get_reverse_scaning_weights(np.array(filter_PWM_human[k]))))} for k in filter_PWM_human.keys()]
        #forward and reverse scanning with same matrix
        forward_and_reverse_direction_filter_list=[{k:np.dstack((filter_PWM_human[k],filter_PWM_human[k]))} for k in filter_PWM_human.keys()]
        forward_and_reverse_direction_filter_dict=dict(j for i in forward_and_reverse_direction_filter_list for j in i.items())
        unequefilter_shape=pa.get_dummies([filter_PWM_human[k].shape for k in filter_PWM_human])
        TF_with_common_dimmention=[{i:list(unequefilter_shape.loc[list(unequefilter_shape[i]==1),:].index)} for i in unequefilter_shape.columns]
        filterr={}
        for i in TF_with_common_dimmention:
            #print(list(i.keys()))
            aa=[list(forward_and_reverse_direction_filter_list[i].keys()) for i in list(i.values())[0]]
            aa=sum(aa,[])
            #print(aa)
            xx=[forward_and_reverse_direction_filter_dict[j] for j in aa]
            #print(xx)
            xxx=np.stack(xx,axis=-1)
            #xxx=xxx.reshape(xxx.shape[1],xxx.shape[2],xxx.shape[3],xxx.shape[0])
            filterr["-".join(aa)]=xxx
            
            
        weights=[v for k,v in filterr.items()]
        protein_names=[k for k,v in filterr.items()]
        protein_names=[n.split("-") for n in protein_names]
        
        return (weights,protein_names)
        
    def get_sequenceBy_Id(self, EnsemblID,content="application/json",expand_5prime=2000, formatt="fasta",
                              species="homo_sapiens",typee="genomic"):
        server = "http://rest.ensembl.org"
        ext="/sequence/id/"+EnsemblID+"?expand_5prime="+str(expand_5prime)+";format="+formatt+";species="+species+";type="+typee
        r = requests.get(server+ext, headers={"Content-Type" : content})
        _=r
        if not r.ok:
            r.raise_for_status()
            sys.exit()
            
        return(r.json()['seq'][0:int(expand_5prime)+2000])
    
    def seq_to3Darray(self, sequence):
        seq3Darray=pa.get_dummies(list(sequence))
        myseq=Seq(sequence)
        myseq=str(myseq.reverse_complement())
        reverseseq3Darray=pa.get_dummies(list(myseq))
        array3D=np.dstack((seq3Darray,reverseseq3Darray))
        return array3D
    
    def get_seq_by_name(self, target_genes):
        promoter_inhancer_sequence=list(map(self.get_sequenceBy_Id, target_genes))
        threeD_sequence=list(map(self.seq_to3Darray, promoter_inhancer_sequence))
        input_for_convolutional_scan=np.stack((threeD_sequence)).astype('float32')
        return input_for_convolutional_scan
    
    def from_2DtoSeq(self, twoD_seq):
        indToSeq={0:"A",1:"C",2:"G",3:"T"} 
        seq=str(''.join([indToSeq[i] for i in np.argmax(twoD_seq, axis=1)]))
        return seq
    
    def conv_single_step(self, seq_slice, W):
        s = seq_slice*W
        # Sum over all entries of the volume s.
        Z = np.sum(s)
        return Z
    
    def conv_single_filter(self, seq,W,stridev,strideh):
        (fv, fh, n_C_prev, n_C) = W.shape

        m=seq.shape[0]
        pad=0
        n_H = int((((seq.shape[1]-fv)+(2*pad))/stridev)+1)
        n_W = int((((seq.shape[2]-fh)+(2*pad))/strideh)+1)
        Z = np.zeros((m, n_H, n_W ,n_C_prev, n_C))
        for i in range(m):
            for h in range(int(n_H)):
                vert_start = h*stridev
                vert_end = stridev*h+fv
                for w in range(int(n_W)):
                    horiz_start = w*strideh
                    horiz_end = strideh*w+fh
                    for c in range(int(n_C)): 
                        a_slice_prev = seq[i,vert_start:vert_end,horiz_start:horiz_end,:]
                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        for d in range(n_C_prev):
                            Z[i, h, w,d, c] = self.conv_single_step(a_slice_prev[:,:,d], W[:,:,d,c])

        Z=self.softmax(Z)                    
        return Z
    
    
    def conv_total_filter(self, Weights, seqs,stridev,strideh):
        return [self.conv_single_filter(seqs,i,stridev,strideh) for i in Weights]
    
    def single_sigmoid_pool(self, motif_score):
        n=sum(motif_score>.5)
        score=[motif_score[i] for i in np.argsort(motif_score)[::-1][:n]]
        index=[j for j in np.argsort(motif_score)[::-1][:n]]
        sigmoid_pooled=dict(zip(index, score))
        sigmoid_pooled=sorted(sigmoid_pooled.items(), key=lambda x: x[1])[::-1]
        return sigmoid_pooled
    
    def total_sigmoid_pool(self, z):
        sigmoid_pooled_motifs=[]
        for i in range(z.shape[0]):
            proteins=[]
            for k in range(z.shape[4]):
                strands=[]
                for j in range(z.shape[3]):
                    strands.append(self.single_sigmoid_pool(z[i,:,0,j,k]))
                proteins.append(strands)
            sigmoid_pooled_motifs.append(proteins)
        #return np.stack(sigmoid_pooled_motifs)
        return np.array(sigmoid_pooled_motifs)
    def extract_binding_sites_per_protein(self, seq, motif_start, motif_leng):
        return seq[motif_start:motif_start+motif_leng]
    
    def getScore(self, seq, weights):
        NtoInd={"A":0,"C":1,"G":2,"T":3}
        cost=0
        for i in range(len(seq)):
            cost+=weights[i,NtoInd[seq[i]]]
        return cost
    def motifs(self, seqs, mot, weights, protein_names):
        Motifs=[]
        for m in range(len(mot)):
            motifs_by_seq=[]
            for z in range(mot[m].shape[0]):
                motifs_by_protein=[]
                for i in range(mot[m].shape[1]):
                    motifs_by_strand=[]
                    for j in range(mot[m].shape[2]):
                        seqq=[self.extract_binding_sites_per_protein(self.from_2DtoSeq(seqs[z,:,:,j]),l,weights[m].shape[0]) for l in list(pa.DataFrame(mot[m][z,i,j])[0])]

                        score=[self.getScore(k,weights[m][:,:,j,i]) for k in seqq]
                        #coordinate=[{p:p+weights[m]} for p in list(pa.DataFrame(mot[m][z,i,j])[0])]
                        scor_mat={"motif":seqq, "PWM_score":score,"sigmoid_score":list(pa.DataFrame(mot[m][z,i,j])[1]), "protein":protein_names[m][i], "strand":j, "input_Sequence":z, "best_threshold":sum(np.max(weights[m][:,:,j,i], axis=1))*.80}
                        motifs_by_strand.append(scor_mat)
                    motifs_by_protein.append(motifs_by_strand)
                motifs_by_seq.append(motifs_by_protein)
            print(m)    
            Motifs.append(np.stack(motifs_by_seq))
        return Motifs
    
    def flatten_motif(self, xc):
        mymotifs=[]
        for i in range(len(xc)):
            for j in range(xc[i].shape[0]):
                for k in range(xc[i].shape[1]):
                    for z in range(xc[i].shape[2]):
                        if(not pa.DataFrame(xc[i][j,k,z]).sort_values(["PWM_score"], ascending=[0]).loc[list(pa.DataFrame(xc[i][j,k,z]).sort_values(["PWM_score"], ascending=[0])["PWM_score"]>pa.DataFrame(xc[i][j,k,z]).sort_values(["PWM_score"], ascending=[0])["best_threshold"]),:].empty):
                            mymotifs.append(pa.DataFrame(xc[i][j,k,z]).sort_values(["PWM_score"], ascending=[0]).loc[list(pa.DataFrame(xc[i][j,k,z]).sort_values(["PWM_score"], ascending=[0])["PWM_score"]>pa.DataFrame(xc[i][j,k,z]).sort_values(["PWM_score"], ascending=[0])["best_threshold"]),:])

        return pa.concat(mymotifs)
    
    def proteins_motif(self,all_bindings, list_of_prot):
        return [{i:list(all_bindings.loc[list(all_bindings["protein"]==i),"motif"])} for i in list_of_prot]
    
    def filter_by_Sequence_and_strand(self, all_bindings, seq, strand):
         return all_bindings[(all_bindings["input_Sequence"]==seq) & (all_bindings["strand"]==strand)]
    
    def find_all_seq(self, st, substr, start_pos=0, accum=[]):
        ix = st.find(substr, start_pos)
        if ix == -1:
            return accum
        return self.find_all_seq(st, substr, start_pos=ix + 1, accum=accum + [ix])
    
    def get_coordinate(self, seq1, motif):
        return [{i:i+len(motif)} for i in self.find_all_seq(seq1, motif)]

    def get_multiple_coordinate(self, seq1,list_of_motifs):
        motifs_cord=[self.get_coordinate(seq1, i) for i in list_of_motifs]
        motifs_cord_list=sum(motifs_cord, [])
        result = {}
        for d in motifs_cord_list:
            result.update(d)
        return result
    
    def colorTextsingle(self, seq, k, v, color):
        seq=seq[:k]+'{}'+seq[k:v]+'{}'+seq[v:]
        seq=seq.format(color,'\033[0m')
        return seq
    
    def SeqWithMotifs(self, seq,mots,col):
        shifter=0
        for i in range(len(sorted(mots))):
            if(i==0):
                seq=self.colorTextsingle(seq,sorted(mots)[i],mots[sorted(mots)[i]],col)
                shifter+=(len(col)+len("\033[0m"))
                #print("1")
                #print(shifter)
            else:
                if(sorted(mots)[i]>sorted(mots)[i-1] and sorted(mots)[i] < mots[sorted(mots)[i-1]]):
                    #print("yes")
                    #print(seq)

                    seq=seq.replace(seq[mots[sorted(mots)[i-1]]+(shifter-len("\033[0m")):mots[sorted(mots)[i-1]]+shifter],"")
                    #print(seq)
                    temp=seq[:sorted(mots)[i]+shifter-len("\033[0m")]+"\033[0m"
                    #print(temp, seq[sorted(mots)[i]+shifter-len("\033[0m"):])
                    seq=temp+seq[sorted(mots)[i]+shifter-len("\033[0m"):]                
                    #print(seq)
                    #print(seq[sorted(mots)[i]+shifter:mots[sorted(mots)[i]]+shifter+len(col[i])])
                    seq=self.colorTextsingle(seq,sorted(mots)[i]+shifter,mots[sorted(mots)[i]]+shifter, col)
                    #print(seq)
                    shifter+=(len(col[i])+len("\033[0m"))
                    #print(i)
                else:
                    seq=self.colorTextsingle(seq,sorted(mots)[i]+shifter,mots[sorted(mots)[i]]+shifter, col)
                    #print("last")
                    shifter+=(len(col)+len("\033[0m"))
        return seq
    
    def get_motif_by_protein(self, seq, pro_mo, colors):
        col_cod=0
        for dct in pro_mo:
            for k,v in dct.items():
                coor=self.get_multiple_coordinate(seq,v)
                #print(coor)
                seq=self.SeqWithMotifs(seq,coor,colors[col_cod])
            col_cod+=1
        return seq